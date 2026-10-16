"""FastAPI HTTP sınırı; ham istek metnini loglamaz veya kalıcılaştırmaz."""

from __future__ import annotations

import asyncio
import json
import logging
import os

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from .gateway import DemoProvider, Gateway, OpenAICompatibleProvider, UpstreamUnavailable
from .gazetteer import FacilityRecognizer, LocationGazetteer
from .masking import process_demo
from .ner import DetectionPipeline, HuggingFaceNERRecognizer
from .policy import Policy
from .rules_ner import RuleEntityRecognizer

MAX_TEXT_LENGTH = 20_000
MAX_REQUEST_BYTES = 100_000
app = FastAPI(title="Türkçe LLM DLP", version="0.1.0")


class RequestBodyLimitMiddleware:
    """Bound request buffering before JSON parsing or Pydantic validation."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_security_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                existing = {name.lower() for name, _ in headers}
                if b"cache-control" not in existing:
                    headers.append((b"cache-control", b"no-store"))
                if b"x-content-type-options" not in existing:
                    headers.append((b"x-content-type-options", b"nosniff"))
                message = {**message, "headers": headers}
            await send(message)

        body = bytearray()
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                response = JSONResponse(
                    status_code=413, content={"detail": "İstek gövdesi izin verilen boyutu aşıyor."}
                )
                await response(scope, receive, send_with_security_headers)
                return
            more_body = message.get("more_body", False)

        sent = False

        async def replay_receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return {"type": "http.disconnect"}

        await self.app(scope, replay_receive, send_with_security_headers)


app.add_middleware(RequestBodyLimitMiddleware, max_bytes=MAX_REQUEST_BYTES)


logger = logging.getLogger(__name__)


def _configured_policy() -> Policy:
    policy_path = os.environ.get("TURKCE_DLP_POLICY")
    # A malformed configured policy prevents startup; silently falling back
    # could weaken an organization's intended block rules.
    return Policy.from_yaml(policy_path) if policy_path else Policy()


def _rule_recognizers() -> list:
    # Kural tabanlı kurum/adres tanıyıcı modelden bağımsız çalışır; TURKCE_DLP_RULE_NER=0 ile kapatılabilir.
    # Yer adı sözlüğü (il/ilçe/mahalle/ülke) model öncelikli boşluk doldurur; TURKCE_DLP_GAZETTEER=0 ile kapatılır.
    recognizers: list = []
    if os.environ.get("TURKCE_DLP_RULE_NER", "1").strip() != "0":
        recognizers.append(RuleEntityRecognizer())
    if os.environ.get("TURKCE_DLP_GAZETTEER", "1").strip() != "0":
        recognizers.append(LocationGazetteer())
        recognizers.append(FacilityRecognizer())
    return recognizers


def _configured_detector() -> DetectionPipeline:
    model_dir = os.environ.get("TURKCE_DLP_NER_MODEL_DIR", "").strip()
    if not model_dir:
        return DetectionPipeline(_rule_recognizers())
    try:
        device = int(os.environ.get("TURKCE_DLP_NER_DEVICE", "-1"))
        startup_timeout = float(os.environ.get("TURKCE_DLP_NER_STARTUP_TIMEOUT_SECONDS", "60"))
        inference_timeout = float(os.environ.get("TURKCE_DLP_NER_TIMEOUT_SECONDS", "2"))
        recognizer = HuggingFaceNERRecognizer.from_local_model(
            model_dir,
            device=device,
            startup_timeout_seconds=startup_timeout,
            inference_timeout_seconds=inference_timeout,
        )
    except Exception:
        # Keep service startup available with deterministic rules when an
        # optional local model is absent or cannot be loaded. Never log paths
        # that could contain identifying workspace details or request text.
        logger.warning("Yerel NER modeli yüklenemedi; kural tabanlı tespit ile devam ediliyor.")
        return DetectionPipeline(_rule_recognizers())
    return DetectionPipeline([*_rule_recognizers(), recognizer])


def _configured_provider():
    kind = os.environ.get("TURKCE_DLP_UPSTREAM", "demo").lower()
    if kind == "demo":
        return DemoProvider()
    if kind == "openai_compatible":
        timeout = float(os.environ.get("TURKCE_DLP_UPSTREAM_TIMEOUT", "30"))
        return OpenAICompatibleProvider(
            api_key=os.environ.get("TURKCE_DLP_UPSTREAM_API_KEY", ""),
            model=os.environ.get("TURKCE_DLP_UPSTREAM_MODEL", ""),
            endpoint=os.environ.get("TURKCE_DLP_UPSTREAM_URL", "https://api.openai.com/v1/chat/completions"),
            timeout_seconds=timeout,
        )
    raise ValueError("TURKCE_DLP_UPSTREAM demo veya openai_compatible olmalı.")


detection_pipeline = _configured_detector()
gateway = Gateway(_configured_provider(), _configured_policy(), detector=detection_pipeline.detect)


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Metin boş olamaz.")
        return value


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request, exc: RequestValidationError):
    # Do not echo input values in validation responses.
    errors = [
        {
            "loc": error.get("loc", []),
            "msg": error.get("msg", "Geçersiz istek"),
            "type": error.get("type", "value_error"),
        }
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health() -> dict:
    # Provider mode is intentionally omitted: the configured gateway may send
    # masked prompts to an external service, so claiming local-only is unsafe.
    return {"status": "ok"}


@app.post("/v1/detect")
def detect_route(request: TextRequest) -> dict:
    # Deliberately return only categories/offsets; never return matched values.
    matches = detection_pipeline.detect(request.text)
    return {"detections": [{"type": d.type, "start": d.start, "end": d.end} for d in matches], "count": len(matches)}


@app.post("/v1/process")
def process_route(request: TextRequest) -> dict:
    return process_demo(request.text)


@app.post("/v1/gateway")
async def gateway_route(request: TextRequest):
    try:
        result = await gateway.process(request.text)
    except UpstreamUnavailable:
        return JSONResponse(status_code=502, content={"detail": "Upstream sağlayıcı isteği tamamlayamadı."})
    if result["blocked"]:
        return JSONResponse(status_code=403, content=result)
    return result


@app.post("/v1/gateway/stream")
async def gateway_stream_route(request: TextRequest):
    prepared = await asyncio.to_thread(gateway.prepare, request.text)
    if prepared.blocked:
        return JSONResponse(
            status_code=403,
            content={"blocked": True, "detections": prepared.detections, "count": prepared.count},
        )

    async def events():
        try:
            async for fragment in gateway.stream_prepared(prepared):
                # JSON encoding keeps newlines and control characters inside
                # an SSE event and avoids treating them as protocol fields.
                yield f"data: {json.dumps({'delta': fragment}, ensure_ascii=False)}\n\n"
        except UpstreamUnavailable:
            yield 'event: error\ndata: {"detail":"Upstream sağlayıcı akışı tamamlayamadı."}\n\n'
        else:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )
