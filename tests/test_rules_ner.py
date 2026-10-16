import pytest

from turkce_llm_dlp.rules_ner import RuleEntityRecognizer

R = RuleEntityRecognizer()


def found(text):
    return [(s.type, text[s.start : s.end]) for s in R.recognize(text)]


def test_formal_address_with_unit_and_city():
    text = "Siparişiniz Yenişehir Mah. Lale Sk. No:14 D:3 Pendik/İstanbul adresine teslim edilecektir."
    assert found(text) == [("ADDRESS", "Yenişehir Mah. Lale Sk. No:14 D:3 Pendik/İstanbul")]


def test_chat_style_lowercase_ascii_address():
    text = "paketi karsiyaka mah 5011 sk no 12 d 3 ortahisar trabzon a birak"
    assert found(text) == [("ADDRESS", "karsiyaka mah 5011 sk no 12 d 3 ortahisar trabzon")]


def test_suffix_after_apostrophe_stays_outside():
    text = "Kargo Atatürk Cad. No:5 Kırşehir'e gitti."
    assert found(text) == [("ADDRESS", "Atatürk Cad. No:5 Kırşehir")]


def test_number_without_street_context_is_not_address():
    for text in ("No:5 yazan kapıyı bulamadık.", "Fatura bilgisi, vergi no 1234567890.", "Oda no 12'de toplanıyoruz."):
        assert [t for t, _ in found(text)] == []


def test_two_addresses_separated_by_semicolon():
    text = "Teslimat: Hürriyet Mah. Barış Sk. No:2 Bağcılar; Fatih Mah. Erguvan Sk. No:17/1 Sancaktepe"
    assert found(text) == [
        ("ADDRESS", "Hürriyet Mah. Barış Sk. No:2 Bağcılar"),
        ("ADDRESS", "Fatih Mah. Erguvan Sk. No:17/1 Sancaktepe"),
    ]


def test_org_suffixes_and_chained_heads():
    assert found("Ekte Yıldız Yazılım Ltd. Şti. ile sözleşme var.") == [("ORG", "Yıldız Yazılım Ltd. Şti.")]
    assert found("Hasta Gazi Üniversitesi Hastanesi'ne sevk edildi.") == [("ORG", "Gazi Üniversitesi Hastanesi")]
    assert found("Merhaba, Kadıköy Belediyesi yazısı geldi.") == [("ORG", "Kadıköy Belediyesi")]


def test_org_needs_a_name_and_lowercase_as_is_not_company():
    assert found("Bankası belli değil, üniversitesi de.") == []
    assert found("Things such as Kadıköy as well") == []


def test_pipeline_precedence_rule_address_wins_model_org_wins():
    from turkce_llm_dlp.ner import DetectionPipeline, EntitySpan

    text = "Kadıköy Belediyesi Fen İşleri Müdürlüğü'ne, kargo karsiyaka mah 5011 sk no 12 trabzon a bırakıldı"
    addr_start = text.index("karsiyaka")

    class Model:
        def recognize(self, _text):
            # Model kurumu doğru, adresi taşırarak buluyor ("... trabzon a").
            return [
                EntitySpan("ORG", 0, len("Kadıköy Belediyesi")),
                EntitySpan("ADDRESS", addr_start, text.index(" bırakıldı")),
            ]

    found = [(d.type, d.value) for d in DetectionPipeline([RuleEntityRecognizer(), Model()]).detect(text)]
    # Model "Kadıköy Belediyesi" buldu; anotasyon kuralı gereği alt birim de kuruma katılır.
    assert ("ORG", "Kadıköy Belediyesi Fen İşleri Müdürlüğü") in found
    assert ("ADDRESS", "karsiyaka mah 5011 sk no 12 trabzon") in found  # ayrı yazılmış "a" eki dışarıda


def test_rule_and_model_addresses_are_unioned_but_not_into_tckn():
    from turkce_llm_dlp.ner import DetectionPipeline, EntitySpan

    text = "kargo çınar sokak no 12, kadıköy, istanbul; tc 10000000146 tel yok"
    start = text.index("çınar")

    class Model:
        def recognize(self, _text):
            # Model il adını da kapsıyor, ama TCKN'ye kadar taşıyor
            return [EntitySpan("ADDRESS", start, text.index(" tel"))]

    found = [(d.type, d.value) for d in DetectionPipeline([RuleEntityRecognizer(), Model()]).detect(text)]
    assert ("ADDRESS", "çınar sokak no 12, kadıköy, istanbul") in found
    assert any(t == "TCKN" for t, _ in found)
    old = DetectionPipeline([RuleEntityRecognizer(), Model()], address_union=False).detect(text)
    assert [d.type for d in old].count("ADDRESS") == 1


def test_facility_name_joins_following_address():
    from turkce_llm_dlp.gazetteer import LocationGazetteer
    from turkce_llm_dlp.ner import DetectionPipeline

    text = "yeni adresim Gül Apartmanı, Kiraz Sokak No 21, Kayseri"
    found = [(d.type, d.value) for d in DetectionPipeline([RuleEntityRecognizer(), LocationGazetteer()]).detect(text)]
    assert found == [("ADDRESS", "Gül Apartmanı, Kiraz Sokak No 21, Kayseri")]


def test_rule_org_fills_gap_when_model_finds_nothing():
    from turkce_llm_dlp.ner import DetectionPipeline

    found = [
        (d.type, d.value)
        for d in DetectionPipeline([RuleEntityRecognizer()]).detect("Ekte Toros Kimya A.Ş. teklifi var.")
    ]
    assert found == [("ORG", "Toros Kimya A.Ş.")]


def test_implausible_model_address_fragments_are_dropped():
    from turkce_llm_dlp.ner import DetectionPipeline, EntitySpan, plausible_address

    assert plausible_address("Lale Sk. No:5")
    assert plausible_address("kazım karabekir mah 1234 sk no 6")
    assert not plausible_address(": 5")
    assert not plausible_address("blok")
    text = "Kat : 5 , C blok girişi"

    class Model:
        def recognize(self, _text):
            return [EntitySpan("ADDRESS", 4, 7), EntitySpan("ADDRESS", text.index("blok"), text.index("blok") + 4)]

    assert DetectionPipeline([Model()]).detect(text) == []


def test_non_address_entities_need_a_real_word():
    from turkce_llm_dlp.ner import has_name_word

    assert has_name_word("Ali") and has_name_word("ODTÜ") and has_name_word("A101")
    assert not has_name_word("No : 5") and not has_name_word("7") and not has_name_word(":")


def test_org_compounds_city_prefix_and_subunit_are_merged():
    from turkce_llm_dlp.ner import DetectionPipeline, EntitySpan

    def run(text, spans):
        class Model:
            def recognize(self, _text):
                return [EntitySpan(k, text.index(v), text.index(v) + len(v)) for k, v in spans]

        return [(d.type, d.value) for d in DetectionPipeline([Model()]).detect(text)]

    assert run(
        "Fatura Antalya Kurumlar Vergi Dairesi adına kesildi.",
        [("LOCATION", "Antalya"), ("ORG", "Kurumlar Vergi Dairesi")],
    ) == [("ORG", "Antalya Kurumlar Vergi Dairesi")]
    assert run(
        "Yıldız Teknik Üniversitesi Makine Fakültesi bünyesinde",
        [("ORG", "Yıldız Teknik Üniversitesi")],
    ) == [("ORG", "Yıldız Teknik Üniversitesi Makine Fakültesi")]
    assert run(
        "İlgi: Sosyal Güvenlik Kurumu Konya İl Müdürlüğü'nün yazısı.",
        [("ORG", "Sosyal Güvenlik Kurumu"), ("LOCATION", "Konya")],
    ) == [("ORG", "Sosyal Güvenlik Kurumu Konya İl Müdürlüğü")]
    # Kurumsal sözcükle bitmeyen devam birleştirilmez; şehir + marka da birleşmez.
    assert run("Ankara'da Ankara Ticaret Odası", [("LOCATION", "Ankara'da"), ("ORG", "Ankara Ticaret Odası")]) == [
        ("LOCATION", "Ankara"),  # kesme işaretli ek varlığa dahil değil
        ("ORG", "Ankara Ticaret Odası"),
    ]
    assert run("Ankara Getir deposu", [("LOCATION", "Ankara"), ("ORG", "Getir")]) == [
        ("LOCATION", "Ankara"),
        ("ORG", "Getir"),
    ]
    assert run("Aras Kargo şubesi kapalı", [("ORG", "Aras Kargo")]) == [("ORG", "Aras Kargo")]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hey benim adım ırmak ve 10018370224", ["ırmak"]),
        ("benim adım ırmak yaprak", ["ırmak yaprak"]),
        ("merhaba ben ırmak", ["ırmak"]),
        ("ismim Zeynep Kaya, öğrenciyim", ["Zeynep Kaya"]),
        ("Ad Soyad: Ali Kaya", ["Ali Kaya"]),
        ("Ad Soyad: Tuğba Şen", ["Tuğba Şen"]),
        ("İsim: Nazife Arslantürk", ["Nazife Arslantürk"]),
        ("ben Mehmet Yılmaz, satın alma", ["Mehmet Yılmaz"]),
        ("dosyayı mehmet beye ilettim", ["mehmet"]),
        ("ben doğan grubunu sevmem", ["doğan"]),
        ("ben deniz kenarındayım", []),
        ("ben su içtim", []),
        ("bir adım gerisinde", []),
        ("işim gereği çok kullanıyorum", []),
        ("Ben Samsung Galaxy kullanıyorum", []),
        ("BEN MEMNUN KALDIM", []),
        ("ev hanımı olarak çalışıyor", []),
        ("Veli adı soyadı: Tülay Yazıcı\nÖğrencinin okulu", ["Tülay Yazıcı"]),
    ],
)
def test_person_cues(text, expected):
    found = [text[s.start : s.end] for s in RuleEntityRecognizer().recognize(text) if s.type == "PERSON"]
    assert found == expected
