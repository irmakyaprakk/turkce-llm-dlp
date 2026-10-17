#!/usr/bin/env python3
"""Mühürlü test 3 (sealed3) hazırlığı: ham mesajlar → etiketleme taslağı → mühür → doğrulama.

Akış (talimat: docs/sealed3-talimat.md):
  1. Her yazarın mesajları `data/ner/manual/sealed3/kaynak/<herhangi-ad>.txt` dosyasına konur; mesajlar boş
     satırla ayrılır. Bu klasör depoya girmez (.gitignore).
  2. `taslak`: mesajlar tek dosyada `yazarN<TAB>mesaj` satırlarına çevrilir (yazar adları gizlenir, satır içi
     yeni satır `\\n` olur). Çıktı: `data/ner/manual/sealed3/sealed3_toplanan.txt`.
  3. Irmak Yaprak bu dosyayı training/ner/ANNOTATION_RULES.md kurallarıyla elle etiketler ([[ETİKET|metin]]).
     Model ya da kural ön-etiketlemesi kullanılmaz (değerlendirilecek sistemin yanlılığı sete taşınmasın).
  4. `muhurle`: biçim denetlenir, eğitim verisiyle cümle çakışması sayılır, SHA-256 `MANIFEST.json`a yazılır.
     Dosya ve manifest commit'lenir; bundan sonra dosyaya bakılarak ayar yapılmaz.
  5. `dogrula`: değerlendirmeden hemen önce dosyanın mühürden beri değişmediği denetlenir.

Hiçbir komut cümle metni yazdırmaz; çıktı yalnız sayılardır.

Kullanım:
  python training/ner/prepare_sealed3.py taslak
  python training/ner/prepare_sealed3.py muhurle --against data/ner/processed/augmented-v5/train.jsonl \\
      data/ner/processed/augmented-v5/dev.jsonl
  python training/ner/prepare_sealed3.py dogrula
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.ner.build_realistic_set import parse_line  # noqa: E402
from training.ner.prepare_augmented import key  # noqa: E402

SEALED_DIR = ROOT / "data/ner/manual/sealed3"
ANNOTATED = SEALED_DIR / "sealed3_toplanan.txt"
MANIFEST = SEALED_DIR / "MANIFEST.json"
HEADER = (
    "# Mühürlü test 3 (sealed3). Kurallar: training/ner/ANNOTATION_RULES.md\n"
    "# Kaynak: bağımsız insan yazarlar (arkadaşlar elle yazdı). Etiketleyen: Irmak Yaprak.\n"
    "# Biçim: yazarN<TAB>mesaj. Satır içi \\n = yeni satır. Tüm ad, kurum ve adresler uydurmadır.\n"
    "# v5'in tek bağımsız değerlendirmesinde BİR KEZ kullanılır; bakılarak ayar yapılmaz.\n"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_messages(path: Path) -> list[str]:
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8").replace("\r\n", "\n"))
    return [block.strip() for block in blocks if block.strip()]


def annotated_rows(path: Path) -> list[tuple[int, str, str]]:
    rows = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "\t" not in line:
            sys.exit(f"satır {number}: 'yazarN<TAB>mesaj' biçiminde değil")
        author, sentence = line.split("\t", 1)
        rows.append((number, author, sentence))
    return rows


def cmd_taslak(args) -> None:
    sources = sorted(args.kaynak_dir.glob("*.txt"))
    if not sources:
        sys.exit(f"{args.kaynak_dir} içinde .txt yok")
    if args.out.exists() and not args.force:
        sys.exit(f"{args.out} zaten var (etiketleme kaybolmasın diye üzerine yazılmaz; --force)")
    seen: set[str] = set()
    lines, per_author, duplicates = [], Counter(), 0
    for index, source in enumerate(sources, 1):
        for message in read_messages(source):
            k = key(message)
            if k in seen:
                duplicates += 1
                continue
            seen.add(k)
            if "\t" in message or "[[" in message:
                sys.exit(f"yazar{index}: mesajda sekme ya da '[[' var; elle düzeltin")
            lines.append(f"yazar{index}\t{message.replace(chr(10), chr(92) + 'n')}")
            per_author[f"yazar{index}"] += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(HEADER + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"mesaj": len(lines), "yazar": dict(per_author), "tekrar_atildi": duplicates}, ensure_ascii=False))


def cmd_muhurle(args) -> None:
    rows = annotated_rows(args.file)
    entities, unlabeled, texts = Counter(), 0, set()
    for number, _author, sentence in rows:
        try:
            text, spans = parse_line(sentence.replace("\\n", "\n"))
        except (ValueError, KeyError):
            # Hata mesajı cümle parçası içerebilir: yalnız satır numarası yazılır.
            sys.exit(f"satır {number}: etiket işareti hatalı (ANNOTATION_RULES.md)")
        entities.update(label for _, _, label in spans)
        unlabeled += not spans
        texts.add(key(text))
    overlap = 0
    against = {}
    for path in args.against:
        against[str(path)] = sha256(path)
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                overlap += key(row.get("text") or " ".join(row["tokens"])) in texts
    manifest = {
        "file": str(args.file.relative_to(ROOT)) if args.file.is_relative_to(ROOT) else str(args.file),
        "sha256": sha256(args.file),
        "sealed_at": date.today().isoformat(),
        "rows": len(rows),
        "authors": len({author for _, author, _ in rows}),
        "entities": dict(entities),
        "rows_without_entities": unlabeled,
        "training_overlap": overlap,
        "checked_against": against,
        "source": "bağımsız insan yazarlar (arkadaşlar elle yazdı); etiketleyen Irmak Yaprak",
    }
    if overlap:
        sys.exit(f"eğitim verisiyle {overlap} cümle çakışıyor; mühürlenmedi")
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("rows", "authors", "entities", "training_overlap")}, ensure_ascii=False))


def cmd_dogrula(args) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    path = Path(manifest["file"])
    if sha256(path if path.is_absolute() else ROOT / path) != manifest["sha256"]:
        sys.exit("sealed3 mühürden sonra değişmiş: değerlendirme yapılmamalı")
    print(f"sealed3 mühürlü hâliyle aynı ({manifest['rows']} mesaj, mühür {manifest['sealed_at']})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    taslak = sub.add_parser("taslak", help="ham mesajları etiketleme taslağına çevir")
    taslak.add_argument("--kaynak-dir", type=Path, default=SEALED_DIR / "kaynak")
    taslak.add_argument("--out", type=Path, default=ANNOTATED)
    taslak.add_argument("--force", action="store_true")
    muhurle = sub.add_parser("muhurle", help="etiketli dosyayı denetle ve mühürle")
    muhurle.add_argument("--file", type=Path, default=ANNOTATED)
    muhurle.add_argument("--manifest", type=Path, default=MANIFEST)
    muhurle.add_argument("--against", type=Path, nargs="*", default=[])
    dogrula = sub.add_parser("dogrula", help="mühürden beri değişiklik var mı")
    dogrula.add_argument("--manifest", type=Path, default=MANIFEST)
    args = ap.parse_args()
    {"taslak": cmd_taslak, "muhurle": cmd_muhurle, "dogrula": cmd_dogrula}[args.command](args)


if __name__ == "__main__":
    main()
