"""Anonim denetim günlüğü: hangi türden kaç kişisel veri yakalandı — değerin kendisi asla tutulmaz.

Her kayıt yalnızca şunları içerir: dakikaya yuvarlanmış zaman, kaynak (eklenti / gateway), site adı
(chatgpt.com gibi), sonuç (maskelendi / temiz / engellendi / vazgeçildi) ve tür başına SAYILAR
(bulunan, maskelenen, kullanıcının işaretini kaldırdığı). Metin, değer, değerin özeti (hash) ya da
kullanıcı kimliği yazılmaz: TC kimlik no gibi küçük uzaylı değerlerin özeti bile kaba kuvvetle geri
çözülebileceği için özet de tutulmaz. Kayıtlar saklama süresi dolunca silinir (varsayılan 90 gün).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .policy import ENTITY_TYPES

# leak_detected: arka plandaki tespit ağı onaysız bir gönderim gördü (engellenemedi; yalnız bildirim)
Outcome = Literal["masked", "clean", "blocked", "cancelled", "leak_detected"]
MAX_COUNT = 10_000


class AuditEvent(BaseModel):
    """Eklentinin ya da gateway'in gönderdiği tek bir karar kaydı (yalnızca sayılar)."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["extension", "gateway"]
    site: str = Field(min_length=1, max_length=253, pattern=r"^[a-z0-9.-]+$")
    outcome: Outcome
    detected: dict[str, int] = Field(default_factory=dict)
    masked: dict[str, int] = Field(default_factory=dict)
    unchecked: dict[str, int] = Field(default_factory=dict)

    @field_validator("detected", "masked", "unchecked")
    @classmethod
    def only_type_counts(cls, value: dict[str, int]) -> dict[str, int]:
        cleaned = {}
        for kind, count in value.items():
            if kind not in ENTITY_TYPES:
                raise ValueError("Bilinmeyen veri türü.")
            if not isinstance(count, int) or not 0 <= count <= MAX_COUNT:
                raise ValueError("Sayı 0 ile 10000 arasında olmalı.")
            if count:
                cleaned[kind] = count
        return cleaned


class AuditLog:
    """SQLite tabanlı, iş parçacığı güvenli anonim olay deposu."""

    def __init__(self, path: str | Path, retention_days: int = 90):
        if not 1 <= retention_days <= 3650:
            raise ValueError("Saklama süresi 1-3650 gün olmalı.")
        self.path = Path(path)
        self.retention_days = retention_days
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    site TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    detected TEXT NOT NULL,
                    masked TEXT NOT NULL,
                    unchecked TEXT NOT NULL
                )"""
            )
            db.execute("CREATE INDEX IF NOT EXISTS events_ts ON events(ts)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)

    def record(self, event: AuditEvent, now: float | None = None) -> None:
        ts = int((now if now is not None else time.time()) // 60 * 60)  # dakikaya yuvarla
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT INTO events (ts, source, site, outcome, detected, masked, unchecked) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    ts,
                    event.source,
                    event.site,
                    event.outcome,
                    json.dumps(event.detected, sort_keys=True),
                    json.dumps(event.masked, sort_keys=True),
                    json.dumps(event.unchecked, sort_keys=True),
                ),
            )
            db.execute("DELETE FROM events WHERE ts < ?", (ts - self.retention_days * 86_400,))

    def summary(self, days: int = 30, now: float | None = None, recent: int = 20) -> dict:
        now = now if now is not None else time.time()
        since = int(now - days * 86_400)
        with self._lock, self._connect() as db:
            rows = db.execute(
                "SELECT ts, source, site, outcome, detected, masked, unchecked FROM events WHERE ts >= ? ORDER BY ts",
                (since,),
            ).fetchall()
        detected: Counter = Counter()
        masked: Counter = Counter()
        unchecked: Counter = Counter()
        outcomes: Counter = Counter()
        sites: Counter = Counter()
        daily: dict[str, Counter] = defaultdict(Counter)
        recent_rows = []
        for ts, source, site, outcome, det, msk, unc in rows:
            det, msk, unc = json.loads(det), json.loads(msk), json.loads(unc)
            detected.update(det)
            masked.update(msk)
            unchecked.update(unc)
            outcomes[outcome] += 1
            sites[site] += 1
            day = time.strftime("%Y-%m-%d", time.localtime(ts))
            daily[day]["messages"] += 1
            daily[day]["masked"] += sum(msk.values())
            daily[day]["blocked"] += outcome == "blocked"
            recent_rows.append(
                {"ts": ts, "source": source, "site": site, "outcome": outcome, "masked": msk, "unchecked": unc}
            )
        days_list = []
        for offset in range(days - 1, -1, -1):
            day = time.strftime("%Y-%m-%d", time.localtime(now - offset * 86_400))
            days_list.append({"day": day, **{k: daily[day][k] for k in ("messages", "masked", "blocked")}})
        total_detected = sum(detected.values())
        return {
            "days": days,
            "messages": len(rows),
            "outcomes": dict(outcomes),
            "detected": dict(detected),
            "masked": dict(masked),
            "unchecked": dict(unchecked),
            "false_positive_rate": round(sum(unchecked.values()) / total_detected, 4) if total_detected else None,
            "sites": dict(sites.most_common(10)),
            "daily": days_list,
            "recent": list(reversed(recent_rows[-recent:])),
        }
