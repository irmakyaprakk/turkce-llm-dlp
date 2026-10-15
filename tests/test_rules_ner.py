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
    assert ("ORG", "Kadıköy Belediyesi") in found  # kural ORG'u daha uzun ama model öncelikli
    assert ("ADDRESS", "karsiyaka mah 5011 sk no 12 trabzon") in found  # adreste kural öncelikli


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
