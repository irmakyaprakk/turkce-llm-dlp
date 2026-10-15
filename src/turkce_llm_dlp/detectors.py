"""Kural tabanlı, Türkçe odaklı hassas veri tespiti."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from math import log2

ZERO_WIDTH = frozenset("\u200b\u200c\u200d\ufeff\u2060")


@dataclass(frozen=True, slots=True)
class Detection:
    type: str
    start: int
    end: int
    value: str


def normalized_view(text: str, *, enabled: bool = True) -> tuple[str, list[int]]:
    """Return NFKC text and a normalized-index -> original-index map.

    Normalization is detection-only. The returned offsets let callers replace
    the corresponding span in the original, untouched string.
    """
    chars: list[str] = []
    offsets: list[int] = []
    for index, original in enumerate(text):
        if not enabled:
            chars.append(original)
            offsets.append(index)
            continue
        if original in ZERO_WIDTH:
            continue
        for char in unicodedata.normalize("NFKC", original):
            if char not in ZERO_WIDTH:
                chars.append(char)
                offsets.append(index)
    return "".join(chars), offsets


def _span(match: re.Match[str], offsets: list[int]) -> tuple[int, int]:
    return offsets[match.start()], offsets[match.end() - 1] + 1


def valid_tckn(value: str) -> bool:
    if not re.fullmatch(r"[1-9]\d{10}", value, flags=re.ASCII):
        return False
    digits = [int(digit) for digit in value]
    tenth = (sum(digits[0:9:2]) * 7 - sum(digits[1:8:2])) % 10
    return tenth == digits[9] and sum(digits[:10]) % 10 == digits[10]


def valid_iban(value: str) -> bool:
    compact = re.sub(r"[\s-]", "", value).upper()
    if not re.fullmatch(r"TR\d{24}", compact, flags=re.ASCII):
        return False
    rearranged = compact[4:] + compact[:4]
    remainder = 0
    for char in rearranged:
        expanded = str(ord(char) - 55) if char.isalpha() else char
        for digit in expanded:
            remainder = (remainder * 10 + int(digit)) % 97
    return remainder == 1


def valid_card_number(value: str) -> bool:
    """Validate a 13-19 digit payment-card candidate with the Luhn check."""
    digits = re.sub(r"[\s-]", "", value)
    if not re.fullmatch(r"\d{13,19}", digits, flags=re.ASCII):
        return False
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        digit = int(char)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def shannon_entropy(value: str) -> float:
    """Return Shannon entropy in bits per character."""
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return -sum((count / length) * log2(count / length) for count in counts.values())


PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("CARD", re.compile(r"(?<![A-Z0-9])(?:\d[ -]?){12,18}\d(?![A-Z0-9])", re.IGNORECASE | re.ASCII)),
    ("TCKN", re.compile(r"(?<!\d)\d(?:[\s.\-]*\d){10}(?!\d)", re.ASCII)),
    ("IBAN", re.compile(r"(?<![A-Z0-9])TR(?:[\s-]*\d){24}(?!\d)", re.IGNORECASE | re.ASCII)),
    (
        "EMAIL",
        re.compile(
            r"(?<![A-Z0-9._%+-])[A-Z0-9._%+-]+@[A-Z0-9](?:[A-Z0-9-]*[A-Z0-9])?(?:\.[A-Z0-9](?:[A-Z0-9-]*[A-Z0-9])?)*\.[A-Z]{2,}(?![A-Z0-9-])",
            re.IGNORECASE | re.ASCII,
        ),
    ),
    (
        "SECRET",
        re.compile(
            r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]{8,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|sk_(?:live|test)_[A-Za-z0-9]{12,}|sk-proj-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{35}|glpat-[A-Za-z0-9_-]{20,}|npm_[A-Za-z0-9]{30,}|pypi-[A-Za-z0-9_-]{50,}|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,})(?![A-Za-z0-9_])"
        ),
    ),
)

PASSWORD_ASSIGNMENT = re.compile(
    r"(?i)\b(?:password|passwd|pwd)\b\s*[:=]\s*(?:\"(?P<double>[^\"\r\n]{1,512})\"|"
    r"'(?P<single>[^'\r\n]{1,512})'|(?P<bare>[^\s,;]{1,512}))"
)
CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|token|(?:aws[_-]?)?secret(?:[_-](?:access[_-]?)?key)?|"
    r"client[_-]?secret|credential|private[_-]?key)\b\s*[:=]\s*(?:\"(?P<double>[^\"\r\n]{20,512})\"|"
    r"'(?P<single>[^'\r\n]{20,512})'|(?P<bare>[^\s,;]{20,512}))"
)
PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]{20,}?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)


def detect(text: str, *, normalize: bool = True) -> list[Detection]:
    """Detect sensitive values; ``normalize=False`` is for controlled evaluation only."""
    view, offsets = normalized_view(text, enabled=normalize)
    candidates: list[Detection] = []
    for kind, pattern in PATTERNS:
        for match in pattern.finditer(view):
            token = match.group()
            if kind == "TCKN":
                canonical = re.sub(r"\D", "", token)
                if len(canonical) != 11 or not valid_tckn(canonical):
                    continue
            elif kind == "IBAN" and not valid_iban(token):
                continue
            elif kind == "CARD" and not valid_card_number(token):
                continue
            start, end = _span(match, offsets)
            candidates.append(Detection(kind, start, end, text[start:end]))

    # Password assignments are context-specific and may have low entropy.
    # Other generic credentials need both a sensitive key name and sufficient
    # entropy, reducing false positives from ordinary configuration values.
    for pattern, entropy_required in ((PASSWORD_ASSIGNMENT, False), (CREDENTIAL_ASSIGNMENT, True)):
        for match in pattern.finditer(view):
            group = next(name for name in ("double", "single", "bare") if match.group(name) is not None)
            token = match.group(group)
            if entropy_required and shannon_entropy(token) < 3.5:
                continue
            start = offsets[match.start(group)]
            end = offsets[match.end(group) - 1] + 1
            candidates.append(Detection("SECRET", start, end, text[start:end]))

    for match in PRIVATE_KEY_BLOCK.finditer(view):
        start, end = _span(match, offsets)
        candidates.append(Detection("SECRET", start, end, text[start:end]))

    # Resolve any future/custom-rule collisions deterministically: prefer the
    # longest match, then the earliest match, and never mask a span twice.
    selected: list[Detection] = []
    for item in sorted(candidates, key=lambda d: (d.start, -(d.end - d.start), d.type)):
        if not any(item.start < other.end and item.end > other.start for other in selected):
            selected.append(item)
    return sorted(selected, key=lambda d: d.start)
