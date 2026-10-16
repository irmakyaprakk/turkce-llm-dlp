"""Request-scope maskeleme ve yanıt geri eşleme."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from dataclasses import dataclass, field

from .detectors import Detection, detect

LABELS = {
    "TCKN": "T.C. kimlik no",
    "IBAN": "IBAN",
    "CARD": "Kredi kartı",
    "EMAIL": "E-posta",
    "PHONE": "Telefon",
    "SECRET": "Gizli bilgi",
    "PERSON": "Kişi",
    "ORG": "Kurum",
    "ADDRESS": "Konum/adres",
    "LOCATION": "Yer adı",
}


@dataclass
class MaskingSession:
    """In-memory mapping scoped to one request/response cycle."""

    _value_to_placeholder: dict[tuple[str, str], str] = field(default_factory=dict)
    _placeholder_to_value: dict[str, str] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)
    _source_text: str = field(default="", init=False, repr=False)

    def _placeholder(self, detection: Detection) -> str:
        identity = unicodedata.normalize("NFKC", detection.value)
        identity = "".join(char for char in identity if char not in "\u200b\u200c\u200d\ufeff\u2060")
        if detection.type == "TCKN":
            identity = re.sub(r"\D", "", identity)
        elif detection.type == "IBAN":
            identity = re.sub(r"[\s-]", "", identity).upper()
        elif detection.type == "CARD":
            identity = re.sub(r"[\s-]", "", identity)
        elif detection.type == "PHONE":
            identity = re.sub(r"[^0-9Xx*]", "", identity)[-10:].upper()
        key = (detection.type, identity)
        if key not in self._value_to_placeholder:
            self._counts[detection.type] = self._counts.get(detection.type, 0) + 1
            number = self._counts[detection.type]
            placeholder = f"<{detection.type}_{number}>"
            # Do not reinterpret placeholder-looking user text as a protected
            # value when the upstream response is restored.
            while placeholder in self._source_text:
                number += 1
                placeholder = f"<{detection.type}_{number}>"
            self._counts[detection.type] = number
            self._value_to_placeholder[key] = placeholder
            self._placeholder_to_value[placeholder] = detection.value
        return self._value_to_placeholder[key]

    def mask(self, text: str, detections: Iterable[Detection] | None = None) -> tuple[str, list[dict[str, str]]]:
        self._source_text = text
        detections = list(detect(text) if detections is None else detections)
        output: list[str] = []
        public_detections: list[dict[str, str]] = []
        cursor = 0
        for item in detections:
            placeholder = self._placeholder(item)
            output.extend((text[cursor : item.start], placeholder))
            cursor = item.end
            public_detections.append({"type": item.type, "label": LABELS[item.type], "placeholder": placeholder})
        output.append(text[cursor:])
        self._source_text = ""
        return "".join(output), public_detections

    def restore(self, response: str) -> str:
        # Longest first prevents replacement collisions if placeholder formats
        # are extended later. Only placeholders created in this request match.
        for placeholder in sorted(self._placeholder_to_value, key=len, reverse=True):
            response = response.replace(placeholder, self._placeholder_to_value[placeholder])
        return response

    def restore_chunks(self, chunks: Iterable[str]) -> Iterator[str]:
        """Restore placeholders even when an upstream stream splits them.

        Keeps only a possible placeholder prefix between chunks, bounding
        extra buffering by the longest request-local placeholder length.
        """
        pending = ""
        keys = tuple(self._placeholder_to_value)
        for chunk in chunks:
            pending += chunk
            hold = 0
            for placeholder in keys:
                maximum = min(len(placeholder) - 1, len(pending))
                for size in range(maximum, 0, -1):
                    if pending.endswith(placeholder[:size]):
                        hold = max(hold, size)
                        break
            safe, pending = pending[:-hold] if hold else pending, pending[-hold:] if hold else ""
            if safe:
                yield self.restore(safe)
            # Include placeholders added after the iterator was created only
            # if a caller deliberately mutates the session during streaming.
            keys = tuple(self._placeholder_to_value)
        if pending:
            yield self.restore(pending)

    async def restore_async_chunks(self, chunks: AsyncIterable[str]) -> AsyncIterator[str]:
        """Streaming counterpart for async upstream providers."""
        pending = ""
        keys = tuple(self._placeholder_to_value)
        async for chunk in chunks:
            pending += chunk
            hold = 0
            for placeholder in keys:
                maximum = min(len(placeholder) - 1, len(pending))
                for size in range(maximum, 0, -1):
                    if pending.endswith(placeholder[:size]):
                        hold = max(hold, size)
                        break
            safe, pending = pending[:-hold] if hold else pending, pending[-hold:] if hold else ""
            if safe:
                yield self.restore(safe)
        if pending:
            yield self.restore(pending)


def process_demo(text: str) -> dict:
    session = MaskingSession()
    masked, detections = session.mask(text)
    mock = (
        "İsteği aldım. Maskelenmiş bağlamla oluşturulan demo yanıtı:\n\n" + masked
        if detections
        else "İsteği aldım. Demo kurallarının tanıdığı bir hassas değer bulamadım.\n\n" + text
    )
    return {
        "masked_prompt": masked,
        "mock_response": mock,
        "restored_response": session.restore(mock),
        "detections": detections,
        "count": len(detections),
    }
