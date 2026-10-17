import json
from argparse import Namespace

import pytest

from training.ner import prepare_sealed3 as S


def test_draft_seal_and_verify(tmp_path, capsys):
    kaynak = tmp_path / "kaynak"
    kaynak.mkdir()
    (kaynak / "ali.txt").write_text("selam ben nur kemerde oturuyom\n\nyarın gelirim\nbekle\n", encoding="utf-8")
    (kaynak / "veli.txt").write_text("yarın gelirim\nbekle\n\nfatura adresi lazım\n", encoding="utf-8")
    draft = tmp_path / "sealed3_toplanan.txt"
    S.cmd_taslak(Namespace(kaynak_dir=kaynak, out=draft, force=False))
    report = json.loads(capsys.readouterr().out)
    assert report == {"mesaj": 3, "yazar": {"yazar1": 2, "yazar2": 1}, "tekrar_atildi": 1}
    lines = [line for line in draft.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    assert lines[1] == "yazar1\tyarın gelirim\\nbekle"  # yazar adı gizli, yeni satır kaçışlı
    with pytest.raises(SystemExit):
        S.cmd_taslak(Namespace(kaynak_dir=kaynak, out=draft, force=False))  # etiketleme üzerine yazılmaz

    draft.write_text(
        draft.read_text(encoding="utf-8").replace("ben nur kemerde", "ben [[PER|nur]] [[LOC|kemerde]]"),
        encoding="utf-8",
    )
    manifest = tmp_path / "MANIFEST.json"
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"text": "başka bir cümle", "tokens": [], "ner_tags": []}) + "\n", encoding="utf-8")
    S.cmd_muhurle(Namespace(file=draft, manifest=manifest, against=[train]))
    sealed = json.loads(manifest.read_text(encoding="utf-8"))
    assert sealed["rows"] == 3 and sealed["entities"] == {"PERSON": 1, "LOCATION": 1}
    assert sealed["training_overlap"] == 0
    assert "nur" not in capsys.readouterr().out  # çıktıda cümle metni yok

    S.cmd_dogrula(Namespace(manifest=manifest))
    draft.write_text(draft.read_text(encoding="utf-8") + "yazar1\tsonradan eklendi\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        S.cmd_dogrula(Namespace(manifest=manifest))


def test_seal_refuses_training_overlap_and_broken_markup(tmp_path):
    draft = tmp_path / "s.txt"
    draft.write_text("yazar1\tyarın gelirim\n", encoding="utf-8")
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"text": "Yarın gelirim"}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        S.cmd_muhurle(Namespace(file=draft, manifest=tmp_path / "m.json", against=[train]))
    draft.write_text("yazar1\tben [[PER|nur kemerde oturuyom\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        S.cmd_muhurle(Namespace(file=draft, manifest=tmp_path / "m.json", against=[]))
