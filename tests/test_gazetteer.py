"""Yer adı sözlüğü (LocationGazetteer) testleri. Örnekler geliştirme / mühürlü setten alınmamıştır."""

import pytest

from turkce_llm_dlp.gazetteer import LocationGazetteer, stems
from turkce_llm_dlp.ner import DetectionPipeline, EntitySpan
from turkce_llm_dlp.rules_ner import RuleEntityRecognizer


@pytest.fixture(scope="module")
def gazetteer():
    return LocationGazetteer()


def found(gazetteer, text):
    return [text[s.start : s.end] for s in gazetteer.recognize(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hafta sonu Eskişehir'e gidiyoruz.", ["Eskişehir"]),
        ("paket yarın malatyaya ulaşır", ["malatyaya"]),
        ("kuzenim kayserideki fabrikada çalışıyor", ["kayserideki"]),
        ("toplantı nevsehirde olacak", ["nevsehirde"]),  # ASCII yazım
        ("Rize ili Çayeli ilçesinde", ["Rize", "Çayeli"]),
        ("Gelecek ay Güney Kore'ye uçuyorum.", ["Güney Kore"]),
        ("Ordu'ya taşındılar.", ["Ordu"]),
        ("Kale'de buluşalım mı?", ["Kale"]),
        ("dün akşam Kemerburgaz Mahallesi'nde yürüdük", ["Kemerburgaz Mahallesi"]),
        ("ofis Valikonağı Caddesi'nde", ["Valikonağı Caddesi"]),
    ],
)
def test_finds_places(gazetteer, text, expected):
    assert found(gazetteer, text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "ordu geri çekildi",  # yaygın sözcük, küçük harf
        "Ordu birlikleri sınıra ulaştı",  # büyük harf ama "context": kesme işareti yok
        "Pazar günü açık mıyız?",  # gün adı
        "Aralık'ta tatile çıkacağım",  # ay adı, kesme işaretli de olsa
        "karşı komşu çok gürültülü",  # ASCII katlaması "kars" + "ı" değil
        "kulpu kırık bardak",  # ilçe adıyla aynı sözcük
        "çubuk kraker aldım",
        "Kalem çok güzel yazıyor",  # cümle başı
        "Selim Bey toplantıya katılamayacak",  # kişi adı + hitap
        "Başbakan Adnan Menderes'i ziyaret etti",  # ad + soyad
        "İzmir Adliyesi önünde toplandılar",  # kurum adının parçası
        "ABD'li yatırımcılar geldi",  # sıfat
        "Can'ın doğum günü yarın",  # "Çan" ile ASCII çakışması
        "bu mahallede herkes birbirini tanır",
        "Ana cadde trafiğe kapalı",
        "eski evin arka sokağında",
    ],
)
def test_rejects_ambiguous(gazetteer, text):
    assert found(gazetteer, text) == []


def test_stems_handle_softening():
    assert "bucak" in set(stems("bucaga"))
    assert "izmir" in set(stems("izmirdeki"))


def test_address_rule_wins_over_gazetteer_inside_address():
    text = "Teslimat: Yıldız Mah. Çınar Sk. No:4 Kızıltepe/Mardin, teşekkürler."
    detections = DetectionPipeline([RuleEntityRecognizer(), LocationGazetteer()]).detect(text)
    assert [(d.type, d.value) for d in detections] == [("ADDRESS", "Yıldız Mah. Çınar Sk. No:4 Kızıltepe/Mardin")]


def test_model_wins_over_gazetteer():
    class Model:
        def recognize(self, text):
            return [EntitySpan("PERSON", 0, 6)]  # model "Ceyhan"ı kişi buldu

    text = "Ceyhan aradı mı?"
    detections = DetectionPipeline([Model(), LocationGazetteer()]).detect(text)
    assert [(d.type, d.value) for d in detections] == [("PERSON", "Ceyhan")]
