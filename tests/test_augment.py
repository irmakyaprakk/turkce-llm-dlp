import random

import pytest

from training.ner.augment import (
    AddressParts,
    apply_variant,
    parse_osm_address,
    render_address,
    row_variant,
    suffix,
    to_ascii,
    tokens_and_tags,
    tr_lower,
    tr_upper,
)
from training.ner.prepare_augmented import Filler, key, render_template


@pytest.mark.parametrize(
    ("word", "kind", "expected"),
    [
        ("İzmir", "DA", "de"),
        ("Bursa", "DA", "da"),
        ("Kadıköy", "DA", "de"),
        ("Bartın", "DA", "da"),
        ("Sinop", "DA", "ta"),
        ("Ankara", "DAN", "dan"),
        ("Tokat", "DAN", "tan"),
        ("Ayşe", "E", "ye"),
        ("Elif", "E", "e"),
        ("Murat", "I", "ı"),
        ("Ümit", "I", "i"),
        ("Konya", "I", "yı"),
        ("Ahmet", "IN", "in"),
        ("Ayşe", "IN", "nin"),
        ("Oğuz", "IN", "un"),
        ("Gül", "IN", "ün"),
        ("Bartın", "LA", "la"),
        ("Moda Mahallesi", "DA", "nde"),
        ("Bağdat Caddesi", "E", "ne"),
        ("Gül Sokağı", "I", "nı"),
    ],
)
def test_suffix_vowel_harmony(word, kind, expected):
    assert suffix(word, kind) == expected


def test_turkish_case_and_ascii_preserve_length():
    assert tr_lower("İSTANBUL IĞDIR") == "istanbul ığdır"
    assert tr_upper("istanbul ığdır") == "İSTANBUL IĞDIR"
    assert to_ascii("Şişli Çağrı Göğüş") == "Sisli Cagri Gogus"
    for text in ("İzmir'de IŞIK", "Çağlayan"):
        assert len(tr_lower(text)) == len(text) == len(to_ascii(text))


def test_variants_keep_labels_on_entities():
    text = "Yarın Kadıköy'de Ayşe Yılmaz'la buluşacağım."
    spans = [(6, 13, "LOCATION"), (17, 28, "PERSON")]
    for variant in ("orig", "lower", "upper", "ascii", "lower_ascii", "noapos", "lower_noapos", "chat"):
        new_text, new_spans = apply_variant(text, spans, variant)
        tokens, tags = tokens_and_tags(new_text, new_spans)
        labelled = [(tok, tag) for tok, tag in zip(tokens, tags) if tag != "O"]
        assert [tag for _, tag in labelled] == ["B-LOCATION", "B-PERSON", "I-PERSON"], variant
    chat_text, _ = apply_variant(text, spans, "chat")
    assert chat_text == "yarin kadikoyde ayse yilmazla bulusacagim."


def test_tokens_and_tags_rejects_overlap():
    with pytest.raises(ValueError):
        tokens_and_tags("Ali Veli", [(0, 3, "PERSON"), (0, 8, "PERSON")])


def test_row_variant_keeps_bio_sequence():
    tokens = ["Dün", "Ankara'da", "Mehmet", "Kaya", "ile", "görüştük", "."]
    tags = ["O", "B-LOCATION", "B-PERSON", "I-PERSON", "O", "O", "O"]
    new_tokens, new_tags = row_variant(tokens, tags, "chat")
    assert new_tokens == ["dun", "ankarada", "mehmet", "kaya", "ile", "gorustuk", "."]
    assert new_tags == tags


def test_osm_parse_and_render_never_loses_number():
    parts = parse_osm_address("Lale Sokak No: 14/2, 34000 Örnek Mahallesi")
    assert parts == AddressParts("Lale Sokak", "14/2", "34000", "Örnek Mahallesi")
    assert parts.il == "İstanbul" and parts.mahalle == "Örnek"
    rng = random.Random(0)
    for _ in range(200):
        rendered = render_address(parts, rng)
        assert "14" in rendered
        assert "Lale" in rendered
    assert parse_osm_address("format dışı satır") is None


def test_template_rendering_spans_and_suffix():
    rng = random.Random(1)
    pool = [AddressParts("Lale Sokak", "5", "06000", "Örnek Mahallesi")]
    filler = Filler(rng, pool, {"PERSON": set(), "ORG": set(), "LOCATION": set()})
    text, spans = render_template("{FIRST:E} söyle, {LOC:DA} buluşalım. Adres: {ADDR}", filler)
    assert "'" in text
    labels = sorted(label for _, _, label in spans)
    assert labels == ["ADDRESS", "LOCATION", "PERSON"]
    for start, end, _ in spans:
        assert text[start:end].strip() == text[start:end]


def test_filler_blocks_gold_names():
    rng = random.Random(2)
    pool = [AddressParts("Lale Sokak", "5", "06000", "Örnek Mahallesi")]
    blocked = {key("Ayşe"), key("Mehmet")}
    filler = Filler(rng, pool, {"PERSON": blocked, "ORG": {key("Migros")}, "LOCATION": set()})
    assert "Ayşe" not in filler.first_names and "Mehmet" not in filler.first_names
    assert "Migros" not in filler.orgs


def test_original_ratio_keeps_raw_osm_format():
    parts = AddressParts("Lale Sokak", "5", "34000", "Örnek Mahallesi")
    rng = random.Random(0)
    outputs = [render_address(parts, rng, original_ratio=1.0) for _ in range(3)]
    assert outputs == ["Lale Sokak No: 5, 34000 Örnek Mahallesi"] * 3
