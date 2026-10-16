#!/bin/bash
# Kod güncellendikten sonra servisi yeniden başlatır (kurulu olmalı: servisi_kur.command).
LABEL="com.turkcedlp.service"
if launchctl kickstart -k "gui/$(id -u)/$LABEL" 2>/dev/null; then
  echo "Servis yeniden başlatılıyor..."
  for _ in $(seq 1 60); do curl -sf http://127.0.0.1:8000/health > /dev/null && { echo "Hazır."; break; }; sleep 1; done
else
  echo "Servis kurulu değil. Önce servisi_kur.command dosyasını çalıştırın."
fi
read -r -p "Kapatmak için Enter'a basın..." _
