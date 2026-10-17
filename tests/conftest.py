import os

# Testler kullanıcının gerçek denetim günlüğüne (~/.turkce-dlp) yazmasın; denetim testleri kendi
# geçici veritabanını kurar.
os.environ.setdefault("TURKCE_DLP_AUDIT", "0")
