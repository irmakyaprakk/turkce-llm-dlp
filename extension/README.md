# Chrome eklentisi

Manifest V3 eklentisi ChatGPT ve Claude sayfalarında gönderim öncesi mesaj metnini yerel `http://127.0.0.1:8000/v1/detect` servisine yollar. Tespit edilen span'leri tarayıcı içinde maskeler, orijinal değerleri yalnızca içerik betiğinin belleğinde tutar ve yanıt DOM'una dönen placeholder'ları yerel olarak geri eşler. Ham değerler uzaktaki LLM'e gönderilmez; yerel API'ye gönderilir.

## Yerel yükleme

1. API'yi `python -m turkce_llm_dlp` ile başlat.
2. Chrome'da `chrome://extensions` açıp Developer mode'u etkinleştir.
3. **Load unpacked** seçip bu `extension/` klasörünü göster.
4. Eklentiyi yalnızca `chatgpt.com` veya `claude.ai` sayfasında kullan.

Kontrol servisi kapalıysa, isteğin boyutu sınırı aşıyorsa veya API yanıtı geçersizse eklenti gönderimi durdurup uyarı verir. Popup'tan koruma kapatılabilir; kapalıyken koruma uygulanmaz.

## Sınırlar

ChatGPT/Claude arayüzleri resmi eklenti API'si sunmadığı için gönderim düğmesi ve yanıt DOM'u değişebilir. Bu ilk sürüm, erişilebilir etiket/textarea/contenteditable seçicilerini kullanır; gerçek sitelerde kullanıcı tarafından elle test edilmelidir. Dosya ekleri, görseller, çok mesajlı bağlam ve eklenti kapalıyken gönderim kapsanmaz. Yanıt DOM'u placeholder'ı değiştirebilir veya parçalara ayırabilir; bu durumda geri eşleme garanti edilmez. DLP için eklentiyi tek koruma katmanı sayma.

JavaScript yardımcı fonksiyon testleri `node --test extension/core.test.cjs` ile çalışır.
