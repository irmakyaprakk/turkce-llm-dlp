#!/bin/bash
# Türkçe LLM DLP yerel servisini macOS arka plan servisi (LaunchAgent) olarak kurar.
# Çift tıklayın; bir kez kurulduktan sonra servis her oturum açılışında kendiliğinden başlar,
# çökerse yeniden başlatılır. Terminal açık kalmak zorunda değildir.
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
LABEL="com.turkcedlp.service"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/turkce-dlp.log"
PYTHON="$PROJECT/.venv/bin/python"
MODEL_DIR="${TURKCE_DLP_NER_MODEL_DIR:-$PROJECT/models/ner/v4/augmented_v4_seed42/model}"

echo "Proje klasörü: $PROJECT"
if [ ! -x "$PYTHON" ]; then
  echo "HATA: $PYTHON bulunamadı. Önce projede sanal ortamı kurun:"
  echo "  cd \"$PROJECT\" && python3 -m venv .venv && .venv/bin/pip install -e '.[ner]'"
  read -r -p "Kapatmak için Enter'a basın..." _; exit 1
fi
if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "UYARI: Model bulunamadı ($MODEL_DIR). Servis yalnızca kural tabanlı tespitle çalışacak."
  MODEL_DIR=""
fi

mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
MODEL_ENV=""
if [ -n "$MODEL_DIR" ]; then
  MODEL_ENV="<key>TURKCE_DLP_NER_MODEL_DIR</key><string>$MODEL_DIR</string>"
fi
cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>$PYTHON</string><string>-m</string><string>turkce_llm_dlp</string></array>
  <key>WorkingDirectory</key><string>$PROJECT</string>
  <key>EnvironmentVariables</key>
  <dict>$MODEL_ENV<key>PYTHONUNBUFFERED</key><string>1</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict>
</plist>
PLIST

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "Servis kuruldu; model yükleniyor (ilk açılışta 10-30 sn sürebilir)..."
for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8000/health > /dev/null; then
    echo "Hazır: http://127.0.0.1:8000/health çalışıyor. Bu pencereyi kapatabilirsiniz."
    read -r -p "Kapatmak için Enter'a basın..." _; exit 0
  fi
  sleep 1
done
echo "Servis 60 sn içinde yanıt vermedi. Kayıt: $LOG"
echo "Not: macOS 'Masaüstü klasörüne erişim' izni sorarsa İzin Ver deyin ve bu dosyayı tekrar çalıştırın."
tail -20 "$LOG" 2>/dev/null || true
read -r -p "Kapatmak için Enter'a basın..." _
