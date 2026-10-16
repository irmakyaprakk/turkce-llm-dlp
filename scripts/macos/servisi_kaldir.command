#!/bin/bash
# Arka plan servisini durdurur ve oturum açılışında başlamasını kaldırır.
LABEL="com.turkcedlp.service"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null && echo "Servis durduruldu." || echo "Servis zaten çalışmıyordu."
rm -f "$HOME/Library/LaunchAgents/$LABEL.plist" && echo "Otomatik başlatma kaldırıldı."
read -r -p "Kapatmak için Enter'a basın..." _
