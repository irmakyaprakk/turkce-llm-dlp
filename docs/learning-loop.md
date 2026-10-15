# Öğrenme döngüsü (human-in-the-loop) — tasarım

Amaç: sistemin gerçek kullanımdaki hatalarından, kişisel veriyi kurum dışına çıkarmadan öğrenmesi.

## Akış

1. **Geri bildirim (eklenti):** Kullanıcı maskeleme önizlemesinde "bu kişisel veri değil" (yanlış alarm) ya da
   seçtiği bir ifade için "bunu da maskele" (kaçırma) der. Her geri bildirim isteğe bağlıdır.
2. **Kuyruk (kurum içi):** Geri bildirim, gateway'in çalıştığı **kurum içi** sunucuda şifreli olarak saklanır:
   cümle, işaretlenen span, sistemin tahmini, zaman. Hiçbir veri LLM sağlayıcısına ya da proje dışına gitmez.
   Saklama süresi sınırlıdır (varsayılan 30 gün), süre dolunca otomatik silinir.
3. **Etiketleme (panel):** Yetkili kişi panelde kuyruğu görür, `ANNOTATION_RULES.md`'ye göre onaylar/düzeltir.
   Onaylanan örnek `[[ETİKET|metin]]` biçiminde dışa aktarılır; kişisel değerler gerçek veriyle eğitmemek için
   aynı biçimde uydurma karşılıklarla değiştirilir (bkz. `sealed/sealed_irmak.txt` yöntemi).
4. **Hızlı düzeltme (eğitimsiz):** Tekrarlayan yanlış alarmlar kurum izin listesine, tekrarlayan kaçırmalar
   kurum sözlüğüne eklenir; etkisi anında olur.
5. **Dönemsel yeniden eğitim:** Biriken onaylı örnekler eğitim verisine eklenir, model yeniden eğitilir.

## Kalite kapısı (her döngüde)

- Yeni model; geliştirme seti, eski mühürlü test ve döngüye **hiç girmemiş yeni bir mühürlü parti** üzerinde
  ölçülür. Herhangi birinde token kapsama (DLP metriği) düşerse model yayına alınmaz.
- Geri bildirim örnekleri eğitime girer, değerlendirme setine girmez (sızıntı olmaz).
- Her model sürümü, verisi ve sonuçları sürüm numarasıyla kayıt altına alınır; geri dönüş mümkündür.

## Riskler

- **Kötü niyetli / hatalı geri bildirim:** Doğrudan eğitime girmez; insan onayından geçer.
- **Kişisel veri birikmesi:** Kurum içi, şifreli, süreli; erişim denetim günlüğüne yazılır (KVKK ilkeleri).
- **Seçim yanlılığı:** Yalnız fark edilen hatalar gelir; rastgele örneklenmiş, geri bildirimsiz trafikten de
  (izinle) küçük bir pay etiketlenir.
