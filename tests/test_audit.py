import sqlite3

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from turkce_llm_dlp import api as api_module
from turkce_llm_dlp.api import app
from turkce_llm_dlp.audit import AuditEvent, AuditLog

DAY = 86_400
NOW = 1_760_000_000.0


def event(**overrides):
    data = {
        "source": "extension",
        "site": "chatgpt.com",
        "outcome": "masked",
        "detected": {"PERSON": 2},
        "masked": {"PERSON": 1},
        "unchecked": {"PERSON": 1},
    }
    data.update(overrides)
    return AuditEvent(**data)


def test_record_and_summary(tmp_path):
    log = AuditLog(tmp_path / "a.db")
    log.record(event(), now=NOW)
    log.record(event(outcome="blocked", detected={"TCKN": 1}, masked={}, unchecked={}), now=NOW - DAY)
    log.record(event(outcome="clean", detected={}, masked={}, unchecked={}), now=NOW)
    s = log.summary(days=7, now=NOW)
    assert s["messages"] == 3
    assert s["outcomes"] == {"masked": 1, "blocked": 1, "clean": 1}
    assert s["masked"] == {"PERSON": 1}
    assert s["false_positive_rate"] == round(1 / 3, 4)
    assert len(s["daily"]) == 7 and sum(d["messages"] for d in s["daily"]) == 3
    assert s["recent"][0]["outcome"] in {"masked", "clean"}


def test_timestamps_rounded_and_no_text_columns(tmp_path):
    path = tmp_path / "a.db"
    AuditLog(path).record(event(), now=NOW + 37.9)
    with sqlite3.connect(path) as db:
        (ts,) = db.execute("SELECT ts FROM events").fetchone()
        columns = [row[1] for row in db.execute("PRAGMA table_info(events)")]
    assert ts % 60 == 0
    assert columns == ["id", "ts", "source", "site", "outcome", "detected", "masked", "unchecked"]


def test_retention_deletes_old_events(tmp_path):
    log = AuditLog(tmp_path / "a.db", retention_days=30)
    log.record(event(), now=NOW - 31 * DAY)
    log.record(event(), now=NOW)
    assert log.summary(days=365, now=NOW)["messages"] == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"text": "Ali Kaya"},  # bilinmeyen alan: metin taşınamaz
        {"detected": {"Ali Kaya": 1}},  # değer, tür adı gibi gizlenemez
        {"detected": {"PERSON": -1}},
        {"detected": {"PERSON": 10_001}},
        {"site": "https://chatgpt.com/c/123"},
        {"outcome": "sent"},
    ],
)
def test_event_rejects_anything_but_counts(bad):
    with pytest.raises(ValidationError):
        event(**bad)


@pytest.fixture
def audited(tmp_path, monkeypatch):
    log = AuditLog(tmp_path / "a.db")
    monkeypatch.setattr(api_module, "audit_log", log)
    monkeypatch.delenv("TURKCE_DLP_ADMIN_TOKEN", raising=False)
    return log


def test_event_endpoint_and_summary(audited):
    client = TestClient(app)
    body = {
        "source": "extension",
        "site": "claude.ai",
        "outcome": "masked",
        "detected": {"PHONE": 1},
        "masked": {"PHONE": 1},
    }
    assert client.post("/v1/audit/event", json=body).status_code == 204
    assert client.post("/v1/audit/event", json={**body, "value": "05321234567"}).status_code == 422
    summary = client.get("/v1/audit/summary?days=7").json()
    assert summary["messages"] == 1 and summary["sites"] == {"claude.ai": 1}
    assert summary["policy"]["rules"]["PHONE"] in {"mask", "block", "allow"}


def test_gateway_records_counts_only(audited):
    client = TestClient(app)
    client.post("/v1/gateway", json={"text": "İletişim: kisi@example.com"})
    summary = client.get("/v1/audit/summary").json()
    assert summary["messages"] == 1
    assert summary["recent"][0]["source"] == "gateway"
    assert "kisi@example.com" not in str(summary)


def test_gateway_stream_records_counts_only(audited):
    client = TestClient(app)
    with client.stream("POST", "/v1/gateway/stream", json={"text": "İletişim: kisi@example.com"}) as response:
        assert response.status_code == 200
        "".join(response.iter_text())
    summary = client.get("/v1/audit/summary").json()
    assert summary["messages"] == 1
    assert summary["outcomes"] == {"masked": 1} and summary["masked"] == {"EMAIL": 1}
    assert "kisi@example.com" not in str(summary)


def test_summary_requires_admin_token_when_configured(audited, monkeypatch):
    monkeypatch.setenv("TURKCE_DLP_ADMIN_TOKEN", "gizli-anahtar")
    client = TestClient(app)
    assert client.get("/v1/audit/summary").status_code == 401
    assert client.get("/v1/audit/summary", headers={"X-Admin-Token": "yanlis"}).status_code == 401
    assert client.get("/v1/audit/summary", headers={"X-Admin-Token": "gizli-anahtar"}).status_code == 200


def test_summary_404_when_audit_disabled(monkeypatch):
    monkeypatch.setattr(api_module, "audit_log", None)
    monkeypatch.delenv("TURKCE_DLP_ADMIN_TOKEN", raising=False)
    assert TestClient(app).get("/v1/audit/summary").status_code == 404


def test_panel_served_with_strict_csp():
    client = TestClient(app)
    page = client.get("/panel")
    assert page.status_code == 200 and "Yönetici Paneli" in page.text
    assert "script-src 'self'" in page.headers["content-security-policy"]
    script = client.get("/panel/panel.js")
    assert script.status_code == 200 and script.headers["content-type"].startswith("text/javascript")
    assert "innerHTML" not in script.text


def test_leak_detected_outcome_is_counted(audited):
    client = TestClient(app)
    body = {"source": "extension", "site": "chatgpt.com", "outcome": "leak_detected"}
    assert client.post("/v1/audit/event", json=body).status_code == 204
    assert client.get("/v1/audit/summary?days=1").json()["outcomes"] == {"leak_detected": 1}
