"""Yer adı sözlüğü verisinin (tr_gazetteer.json.gz) yüklenmesi; bağımlılıksız, döngüsel içe aktarmayı önler."""

from __future__ import annotations

import gzip
import json
from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=1)
def load_gazetteer() -> dict:
    with resources.files("turkce_llm_dlp").joinpath("data/tr_gazetteer.json.gz").open("rb") as raw:
        data = json.loads(gzip.decompress(raw.read()).decode("utf-8"))
    return {
        "places": data["places"],
        "units": frozenset(data["unit_names"]),
        "collisions": frozenset(data["lower_collisions"]),
    }


def province_and_district_keys() -> frozenset[str]:
    """Belirsiz olmayan il ve ilçe adlarının anahtarları (adres kuralının sağa genişletmesi için).
    "Zile", "Kale" gibi aynı zamanda sözcük olan adlar dahil edilmez ("... esenyurt zile bas")."""
    return frozenset(
        key
        for key, entry in load_gazetteer()["places"].items()
        if entry["kind"] in ("il", "ilce") and entry["tier"] == "free"
    )
