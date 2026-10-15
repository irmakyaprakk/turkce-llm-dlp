"""Policy enforcement and upstream-provider boundary."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Protocol

import httpx

from .detectors import Detection, detect
from .masking import LABELS, MaskingSession
from .policy import Policy

MAX_PROVIDER_RESPONSE_BYTES = 1_000_000
Detector = Callable[[str], list[Detection]]


class UpstreamProvider(Protocol):
    async def complete(self, prompt: str) -> str:
        """Send a prompt and return provider text. Implementations must not log it."""

    def stream(self, prompt: str) -> AsyncIterator[str]:
        """Yield response text fragments without logging prompt or response."""


@dataclass(slots=True)
class PreparedRequest:
    blocked: bool
    detections: list[dict[str, str]]
    count: int
    session: MaskingSession | None = None
    upstream_prompt: str | None = None
    masked_count: int = 0


class DemoProvider:
    """Local stand-in; useful for exercising the gateway without network calls."""

    async def complete(self, prompt: str) -> str:
        return "Demo sağlayıcı yanıtı:\n\n" + prompt

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        answer = await self.complete(prompt)
        for start in range(0, len(answer), 11):
            yield answer[start : start + 11]


class OpenAICompatibleProvider:
    """Small chat-completions adapter for OpenAI-compatible HTTPS APIs."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        endpoint: str = "https://api.openai.com/v1/chat/completions",
        timeout_seconds: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if not api_key.strip() or not model.strip():
            raise ValueError("Upstream API anahtarı ve model adı gereklidir.")
        parsed = httpx.URL(endpoint)
        local_http = parsed.scheme == "http" and parsed.host in {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme != "https" and not local_http:
            raise ValueError("Upstream HTTPS kullanmalı (yerel testte localhost HTTP kabul edilir).")
        if not 0 < timeout_seconds <= 120:
            raise ValueError("Upstream zaman aşımı 0 ile 120 saniye arasında olmalı.")
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def complete(self, prompt: str) -> str:
        # Per-request client lifetime avoids persistent connection state and
        # guarantees closure. Neither request nor response bodies are logged.
        async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self.transport) as client:
            async with client.stream(
                "POST",
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": [{"role": "user", "content": prompt}], "stream": False},
            ) as response:
                if not 200 <= response.status_code < 300:
                    raise RuntimeError("Upstream HTTP error")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_PROVIDER_RESPONSE_BYTES:
                        raise RuntimeError("Upstream yanıtı boyut sınırını aştı")
        try:
            payload = json.loads(body)
            content = payload["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Upstream yanıt biçimi geçersiz") from exc
        if not isinstance(content, str):
            raise RuntimeError("Upstream yanıt metin değil")
        return content

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        total_bytes = 0
        buffer = bytearray()
        async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self.transport) as client:
            async with client.stream(
                "POST",
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}", "Accept": "text/event-stream"},
                json=request,
            ) as response:
                if not 200 <= response.status_code < 300:
                    raise RuntimeError("Upstream HTTP error")
                async for chunk in response.aiter_bytes():
                    total_bytes += len(chunk)
                    if total_bytes > MAX_PROVIDER_RESPONSE_BYTES:
                        raise RuntimeError("Upstream yanıtı boyut sınırını aştı")
                    buffer.extend(chunk)
                    while b"\n" in buffer:
                        line, _, remainder = buffer.partition(b"\n")
                        buffer = bytearray(remainder)
                        await_chunk = self._event_data(line.rstrip(b"\r"))
                        if await_chunk is not None:
                            if await_chunk == "[DONE]":
                                return
                            yield await_chunk
        if buffer:
            final = self._event_data(bytes(buffer).rstrip(b"\r"))
            if final and final != "[DONE]":
                yield final

    @staticmethod
    def _event_data(line: bytes) -> str | None:
        if not line.startswith(b"data:"):
            return None
        data = line[5:].strip()
        if data == b"[DONE]":
            return "[DONE]"
        try:
            payload = json.loads(data)
            content = payload["choices"][0]["delta"].get("content")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Upstream akış biçimi geçersiz") from exc
        if content is None:
            return None
        if not isinstance(content, str):
            raise RuntimeError("Upstream akış parçası metin değil")
        return content


class UpstreamUnavailable(RuntimeError):
    pass


class Gateway:
    def __init__(self, provider: UpstreamProvider, policy: Policy | None = None, detector: Detector = detect):
        self.provider = provider
        self.policy = policy or Policy()
        self.detector = detector

    def prepare(self, text: str) -> PreparedRequest:
        detected = self.detector(text)
        decisions = self.policy.decide(detected)
        metadata = [{"type": item.type, "label": LABELS[item.type], "action": action} for item, action in decisions]
        if any(action == "block" for _, action in decisions):
            return PreparedRequest(True, metadata, len(detected))

        masked_items = [item for item, action in decisions if action == "mask"]
        session = MaskingSession()
        upstream_prompt, masked_metadata = session.mask(text, masked_items)
        return PreparedRequest(False, metadata, len(detected), session, upstream_prompt, len(masked_metadata))

    async def process(self, text: str) -> dict:
        # NER is synchronous model inference; keep it off the ASGI event loop.
        prepared = await asyncio.to_thread(self.prepare, text)
        if prepared.blocked:
            return {"blocked": True, "detections": prepared.detections, "count": prepared.count}
        assert prepared.session is not None and prepared.upstream_prompt is not None
        try:
            upstream_response = await self.provider.complete(prepared.upstream_prompt)
        except Exception as exc:
            # Never fall back to sending the original prompt or return a raw
            # provider exception that might contain request data.
            raise UpstreamUnavailable("Upstream sağlayıcı isteği tamamlayamadı.") from exc

        return {
            "blocked": False,
            "response": prepared.session.restore(upstream_response),
            "detections": prepared.detections,
            "count": prepared.count,
            "masked_count": prepared.masked_count,
        }

    async def stream_prepared(self, prepared: PreparedRequest) -> AsyncIterator[str]:
        if prepared.blocked or prepared.session is None or prepared.upstream_prompt is None:
            raise ValueError("Engellenmiş istek upstream akışına verilemez.")

        async def safe_stream() -> AsyncIterator[str]:
            try:
                async for fragment in self.provider.stream(prepared.upstream_prompt):
                    yield fragment
            except Exception as exc:
                raise UpstreamUnavailable("Upstream sağlayıcı akışı tamamlayamadı.") from exc

        async for restored in prepared.session.restore_async_chunks(safe_stream()):
            yield restored
