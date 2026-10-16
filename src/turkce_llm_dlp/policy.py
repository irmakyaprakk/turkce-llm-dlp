"""Fail-closed, configuration-backed actions for detected entity types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from .detectors import Detection

Action = Literal["mask", "block", "allow"]
ENTITY_TYPES = frozenset({"TCKN", "IBAN", "CARD", "EMAIL", "PHONE", "SECRET", "PERSON", "ORG", "LOCATION", "ADDRESS"})
VALID_ACTIONS = frozenset({"mask", "block", "allow"})


@dataclass(frozen=True, slots=True)
class Policy:
    """Unknown entity types always fail closed to ``mask``."""

    default_action: Action = "mask"
    rules: dict[str, Action] | None = None

    def __post_init__(self) -> None:
        if self.default_action not in VALID_ACTIONS:
            raise ValueError("Varsayılan politika eylemi mask, block veya allow olmalı.")
        normalized = {key.upper(): value for key, value in (self.rules or {}).items()}
        unknown = set(normalized) - ENTITY_TYPES
        if unknown:
            raise ValueError(f"Bilinmeyen veri türü: {', '.join(sorted(unknown))}")
        invalid = {value for value in normalized.values() if value not in VALID_ACTIONS}
        if invalid:
            raise ValueError(f"Geçersiz politika eylemi: {', '.join(sorted(invalid))}")
        object.__setattr__(self, "rules", normalized)

    def action_for(self, entity_type: str) -> Action:
        normalized = entity_type.upper()
        if normalized not in ENTITY_TYPES:
            return "mask"
        return self.rules.get(normalized, self.default_action)

    @classmethod
    def from_mapping(cls, data: object) -> Policy:
        if not isinstance(data, dict) or set(data) - {"default_action", "rules"}:
            raise ValueError("Politika YAML'i yalnızca default_action ve rules alanlarını içermeli.")
        default = data.get("default_action", "mask")
        rules = data.get("rules", {})
        if not isinstance(default, str) or not isinstance(rules, dict):
            raise ValueError("Politika alanlarının türü geçersiz.")
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in rules.items()):
            raise ValueError("Politika kuralları metin anahtar/değer çiftleri olmalı.")
        return cls(default_action=default, rules=rules)  # type: ignore[arg-type]

    @classmethod
    def from_yaml(cls, path: str | Path) -> Policy:
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_mapping(yaml.safe_load(stream))

    def decide(self, detections: list[Detection]) -> list[tuple[Detection, Action]]:
        return [(item, self.action_for(item.type)) for item in detections]
