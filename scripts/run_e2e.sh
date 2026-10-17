#!/usr/bin/env bash
# Eklenti uçtan uca testi: yerel DLP API'si (modelsiz, kurallar), sahte sohbet sayfası ve Chromium.
set -euo pipefail
cd "$(dirname "$0")/.."
AUDIT_DB="$(mktemp -d)/audit.sqlite3"  # gerçek denetim günlüğüne yazılmaz
PY="${PYTHON:-python}"
[ -x .venv/bin/python ] && [ -z "${PYTHON:-}" ] && PY=.venv/bin/python
# 8000'de çalışan servis (ör. macOS arka plan servisi) varsa testler ona gider ve gerçek günlüğe yazar: dur.
if curl -sf --noproxy "*" http://127.0.0.1:8000/health > /dev/null 2>&1; then
  echo "127.0.0.1:8000 kullanımda. Önce arka plan servisini durdurun:" >&2
  echo "  launchctl bootout gui/\$(id -u)/com.turkcedlp.service" >&2
  exit 1
fi
PYTHONPATH=src TURKCE_DLP_AUDIT_DB="$AUDIT_DB" TURKCE_DLP_POLICY=tests/e2e/policy_e2e.yml "$PY" -m turkce_llm_dlp > /tmp/dlp-api.log 2>&1 &
API=$!
"$PY" tests/e2e/serve_mock.py &
MOCK=$!
trap 'kill $API $MOCK 2>/dev/null || true' EXIT
for _ in $(seq 1 600); do curl -sf --noproxy "*" http://127.0.0.1:8000/health > /dev/null && break; sleep 0.2; done
node tests/e2e/extension.e2e.mjs
