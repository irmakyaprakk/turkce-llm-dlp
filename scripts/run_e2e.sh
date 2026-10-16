#!/usr/bin/env bash
# Eklenti uçtan uca testi: yerel DLP API'si (modelsiz, kurallar), sahte sohbet sayfası ve Chromium.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src TURKCE_DLP_POLICY=tests/e2e/policy_e2e.yml python -m turkce_llm_dlp > /tmp/dlp-api.log 2>&1 &
API=$!
python tests/e2e/serve_mock.py &
MOCK=$!
trap 'kill $API $MOCK 2>/dev/null || true' EXIT
for _ in $(seq 1 50); do curl -sf --noproxy "*" http://127.0.0.1:8000/health > /dev/null && break; sleep 0.2; done
node tests/e2e/extension.e2e.mjs
