#!/usr/bin/env python3
"""Elle yazılmış, satır içi işaretli cümleleri NER jsonl formatına çevirir.

Girdi (annotated.txt), her satır:  kategori<TAB>cümle   (kategori opsiyonel)
İşaretleme:  [[ETİKET|metin]]   ETİKET: PER, ORG, LOC, ADRES
Örnek:  kisa_mesaj<TAB>yarın [[PER|Elif Demir]]'e uğrayacağım
Boş satırlar ve # ile başlayan satırlar atlanır. Satır içi \\n yazılırsa çok satırlı metin olur.

Çıktı: out_dir/dev.jsonl, out_dir/test.jsonl, out_dir/manifest.json
Alanlar mevcut veriyle aynı: id, source, text, tokens, ner_tags.
Tokenizasyon WikiNER kuralıdır: kesme işaretli ek kelimeye bağlı kalır ve token tamamen etiketlenir.

Kullanım:
  python build_realistic_set.py annotated.txt --out-dir data/ner/processed/realistic-v1 \
      --dev-n 50 --against data/ner/processed/real-address-context-v1/train.jsonl \
                           data/ner/processed/real-address-v2/train.jsonl
"""

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

# Sağ taraf training/ner/labels.json içindeki entity_types ile birebir aynı (doğrulandı).
TAG_MAP = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOCATION", "ADRES": "ADDRESS"}

SOURCE = "hand-written-realistic-v1"
SPAN_RE = re.compile(r"\[\[([^\|\[\]]+)\|(.+?)\]\]")
# WikiNER kuralı: "Adana'da", "Buffy'ye" tek token (ek kelimeye bağlı, token'ın tamamı etiketlenir).
TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)


def parse_line(line):
    """Satırı (düz metin, [(başlangıç, bitiş, etiket)]) olarak döndürür."""
    parts, spans, pos, out_len = [], [], 0, 0
    for m in SPAN_RE.finditer(line):
        before = line[pos : m.start()]
        parts.append(before)
        out_len += len(before)
        tag, txt = m.group(1).strip(), m.group(2)
        if tag not in TAG_MAP:
            raise ValueError(f"bilinmeyen etiket '{tag}' (geçerli: {', '.join(TAG_MAP)})")
        if "[[" in txt or "]]" in txt:
            raise ValueError("iç içe işaretleme desteklenmiyor")
        if re.search(r"['’]\w", txt):
            raise ValueError(f"span içinde kesme işareti + ek var: '{txt}' (ek span dışında kalmalı)")
        spans.append((out_len, out_len + len(txt), TAG_MAP[tag]))
        parts.append(txt)
        out_len += len(txt)
        pos = m.end()
    parts.append(line[pos:])
    text = "".join(parts)
    if "[[" in text or "]]" in text:
        raise ValueError("kapanmamış veya hatalı [[...|...]] işareti")
    return text, spans


def label(text, spans):
    """Token'ın ortasında biten span yalnızca kesme işaretinde bitebilir ("Kadıköy" + 'de).
    Token'ın tamamı etiketlenir (WikiNER kuralı); maskelemede span kesme işaretinde kesilir."""
    toks = [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
    tags = ["O"] * len(toks)
    for s, e, tag in spans:
        idx = []
        for i, (_, ts, te) in enumerate(toks):
            if ts < s < te:
                raise ValueError(f"span token ortasında başlıyor: '{text[s:e]}'")
            if ts < e < te and text[e] not in "'’":
                raise ValueError(
                    f"span token ortasında bitiyor (kesme işareti yok): '{text[s:e]}'; "
                    "eki de span'a alın veya kesme işaretiyle ayırın"
                )
            if s <= ts < e:
                idx.append(i)
        if not idx:
            raise ValueError(f"boş span: '{text[s:e]}'")
        tags[idx[0]] = f"B-{tag}"
        for i in idx[1:]:
            tags[i] = f"I-{tag}"
    return [t for t, _, _ in toks], tags


def norm(s):
    return " ".join(s.casefold().split())


def spans_of(tokens, tags, kind="ADDRESS"):
    out, cur = [], []
    for tok, tg in zip(tokens, tags):
        if tg == f"B-{kind}":
            if cur:
                out.append(" ".join(cur))
            cur = [tok]
        elif tg == f"I-{kind}" and cur:
            cur.append(tok)
        else:
            if cur:
                out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return [norm(x) for x in out]


def leak_check(rows, against):
    seen_text, seen_addr = set(), set()
    for p in against:
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            seen_text.add(norm(" ".join(d["tokens"])))
            seen_addr.update(spans_of(d["tokens"], d["ner_tags"]))
    text_hits = [r["text"] for r in rows if norm(" ".join(r["tokens"])) in seen_text]
    addr_hits = [(r["text"], a) for r in rows for a in spans_of(r["tokens"], r["ner_tags"]) if a in seen_addr]
    return text_hits, addr_hits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dev-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--against", nargs="*", default=[], help="sızıntı kontrolü için eğitim/dev jsonl dosyaları")
    args = ap.parse_args()

    rows, errors, seen = [], [], set()
    for no, raw in enumerate(Path(args.input).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cat, sent = (line.split("\t", 1) + [None])[:2] if "\t" in line else ("genel", line)
        try:
            text, spans = parse_line(sent.strip().replace("\\n", "\n"))
            if norm(text) in seen:
                raise ValueError("aynı cümle tekrar ediyor")
            seen.add(norm(text))
            tokens, tags = label(text, spans)
        except ValueError as e:
            errors.append(f"satır {no}: {e}")
            continue
        rows.append(
            {
                "id": "hand-" + hashlib.sha1(text.encode()).hexdigest()[:12],
                "source": SOURCE,
                "category": cat.strip(),
                "text": text,
                "tokens": tokens,
                "ner_tags": tags,
            }
        )
    if errors:
        print("HATALAR:\n  " + "\n  ".join(errors), file=sys.stderr)
        sys.exit(1)
    if args.dev_n >= len(rows):
        sys.exit(f"--dev-n ({args.dev_n}) toplam satırdan ({len(rows)}) küçük olmalı")

    if args.against:
        text_hits, addr_hits = leak_check(rows, args.against)
        print(f"Sızıntı kontrolü: {len(text_hits)} birebir cümle, {len(addr_hits)} aynı adres span'ı")
        for t in text_hits[:5]:
            print("  CÜMLE ÇAKIŞMASI:", t)
        for t, a in addr_hits[:5]:
            print("  ADRES ÇAKIŞMASI:", a, "<-", t)
        if text_hits or addr_hits:
            sys.exit("Sızıntı var (cümle veya adres çakışması); bu örnekleri değiştir.")

    rng = random.Random(args.seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    dev = [rows[i] for i in sorted(order[: args.dev_n])]
    test = [rows[i] for i in sorted(order[args.dev_n :])]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": SOURCE,
        "tag_map": TAG_MAP,
        "seed": args.seed,
        "rows": {},
        "sha256": {},
        "categories": {},
        "entities": {},
    }
    for name, part in (("dev", dev), ("test", test)):
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["rows"][name] = len(part)
        manifest["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["categories"][name] = dict(collections.Counter(r["category"] for r in part))
        manifest["entities"][name] = dict(
            collections.Counter(t[2:] for r in part for t in r["ner_tags"] if t.startswith("B-"))
        )
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("rows", "entities", "sha256")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
