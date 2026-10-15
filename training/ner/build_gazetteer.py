#!/usr/bin/env python3
"""Türkiye yer adı sözlüğünü (gazetteer) üretir: `src/turkce_llm_dlp/data/tr_gazetteer.json.gz`.

Kaynaklar (geliştirme seti ve mühürlü test KULLANILMAZ):
  - data/gazetteer/turkey-neighbourhoods/: PTT posta kodu listesinden il / ilçe / mahalle / köy adları (MIT)
  - data/gazetteer/ulkeler_bolgeler.txt: ülke adları ve Türkiye'nin coğrafi bölge / yöre adları
  - WikiNER eğitim bölümü: hangi yer adının aynı zamanda yaygın sözcük ya da kişi adı olduğunu
    ölçmek için (ör. "ordu", "pazar", "merkez", "Fatih", "Selim")
  - UCI kullanıcı yorumları (etiketsiz, gündelik dil): bir ad küçük harfle büyük harfle olduğundan sık
    yazılıyorsa yer adı değil sözcüktür ("kulp", "çubuk", "sürücü" → "Suruç"+"u", "bulanık")

Her ad için bir eşleşme düzeyi (tier) belirlenir:
  free     — her yazımda eşleşir (küçük harf, ASCII, ekli: "ceyhana", "izmirdeki")
  cap      — yaygın sözcük: yalnızca büyük harfle ve cümle başı değilken ya da kesme işaretli ekle ("Kale'de")
  context  — büyük harfle de çoğunlukla başka anlam ("Of", "Batman", "Saray"): yalnızca kesme işaretli ekle
             ya da "ili/ilçesi" bağlamıyla
  calendar — ay/gün adıyla aynı ("Aralık", "Pazar", "Çarşamba"): yalnızca "ili/ilçesi" bağlamıyla
  person   — kişi adı olarak daha sık geçen: yalnızca kesme işaretli ekle ya da "ili/ilçesi" bağlamıyla
  org      — kurum (çoğunlukla spor kulübü) adı olarak daha sık geçen: kesme işaretli ya da bitişik ekle
             ("Beşiktaş'ta", "besiktasta"), çıplak hâlde değil

Kullanım:
  python training/ner/build_gazetteer.py \
      --wikiner data/ner/processed/person-org-location-v1/train.jsonl \
      --reviews data/ner/raw/uci-turkish-user-reviews/turkish-user-review-dataset.zip
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from turkce_llm_dlp.gazetteer import strip_suffix  # noqa: E402
from turkce_llm_dlp.rules_ner import TOKEN_RE, _key  # noqa: E402

SOURCE = ROOT / "data" / "gazetteer" / "turkey-neighbourhoods"
EXTRA = ROOT / "data" / "gazetteer" / "ulkeler_bolgeler.txt"
OUTPUT = ROOT / "src" / "turkce_llm_dlp" / "data" / "tr_gazetteer.json.gz"

# Ay / gün adlarıyla aynı yazılan ilçeler: büyük harf de anlam ayırt etmez.
CALENDAR = {"Aralık", "Pazar", "Perşembe", "Çarşamba", "Cuma", "Ocak", "Şubat", "Mart", "Nisan", "Mayıs"}
# İstatistikle yakalanmayan, gündelik dilde çok sık sözcük olan ilçe adları (yaygın Türkçe sözcük bilgisi).
COMMON_EXTRA = {
    "Tokat",
    "Bayat",
    "Yapraklı",
    "Kale",
    "Çay",
    "Gölbaşı",
    "Merkez",
    "Yenice",
    "Ova",
    "Of",
    "Zile",
    "Tire",
    "Söke",
    "Susuz",
    "Durağan",
    "Arsin",  # "zil+e", "tire" (-), "sök+e", sıfatlar, "arasında" yazım hatası
}
# Hiçbir zaman tek başına yer adı sayılmayanlar (ilçe adı olarak anlamsız ya da aşırı genel).
DROP = {"Merkez"}


def parse_unit(raw: str) -> tuple[str, str] | None:
    """ "Akarsu Mah" → ("Akarsu", "mahalle"); "Camiliyayla Köyü" → ("Camiliyayla", "koy")."""
    text = re.sub(r"\s+", " ", raw).strip()
    match = re.match(r"^(.+?) (Mah|Köyü|Köy|Beldesi)\b", text)
    if not match:
        return None
    name = match.group(1).strip()
    if not re.search(r"[A-Za-zÇĞİÖŞÜçğıöşü]", name) or len(name) < 3:
        return None
    return name, "mahalle" if match.group(2) == "Mah" else "koy"


def wikiner_stats(path: Path):
    """Ad anahtarı başına (etiket, büyük/küçük) sayıları ve küçük harfli sözcük sıklıkları."""
    by_key: dict[str, Counter] = defaultdict(Counter)
    lower_words: Counter = Counter()
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        for token, tag in zip(row["tokens"], row["ner_tags"]):
            base = re.split(r"['’]", token, maxsplit=1)[0]
            if not base or not base[0].isalpha():
                continue
            kind = tag[2:] if tag != "O" else "O"
            case = "low" if base[0].islower() else "cap"
            by_key[_key(base)][(kind, case)] += 1
            if case == "low" and kind == "O":
                lower_words[_key(token)] += 1
    return by_key, lower_words


def review_case_counts(path: Path, names: set[str]) -> dict[str, Counter]:
    """Yorumlarda her adın (ekli biçimler dahil) küçük / büyük harfli geçiş sayısı."""
    with zipfile.ZipFile(path) as archive:
        member = next(n for n in archive.namelist() if n.endswith(".txt"))
        text = archive.read(member).decode("utf-8", errors="ignore")
    counts: dict[str, Counter] = defaultdict(Counter)
    for match in TOKEN_RE.finditer(text):
        token = match.group()
        base = re.split(r"['’]", token, maxsplit=1)[0]
        if not base or not base[0].isalpha():
            continue
        key = _key(base)
        case = "low" if base[0].islower() else "cap"
        if key in names:
            counts[key][case] += 1
        elif base == token:
            stem = strip_suffix(key, names)
            if stem and len(stem) >= 4:
                counts[stem][case] += 1
    return counts


def tier_for(name: str, stats: Counter, review: Counter | None = None) -> str:
    low_o = stats[("O", "low")]
    cap_o = stats[("O", "cap")]
    loc = stats[("LOCATION", "cap")] + stats[("LOCATION", "low")]
    per = stats[("PERSON", "cap")] + stats[("PERSON", "low")]
    org = stats[("ORG", "cap")] + stats[("ORG", "low")]
    if name in CALENDAR:
        return "calendar"
    if cap_o >= 3 and cap_o > 2 * loc:
        return "context"
    if per > loc:
        return "person"
    if org > 2 * max(loc, 1):
        return "org"
    # Ansiklopedik metinde adın küçük harfle (etiketsiz) bir kez bile geçmesi sözcük olduğunu gösterir
    if low_o >= 1 or name in COMMON_EXTRA:
        return "cap"
    if review is not None and review["low"] >= 3 and review["low"] > 2 * review["cap"]:
        return "cap"
    return "free"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wikiner", type=Path, required=True, help="WikiNER eğitim JSONL'i (tokens, ner_tags)")
    ap.add_argument("--reviews", type=Path, required=True, help="UCI yorum zip'i (gündelik dilde büyük/küçük harf)")
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()

    by_key, lower_words = wikiner_stats(args.wikiner)
    cities = json.loads((SOURCE / "cityNames.json").read_text(encoding="utf-8"))
    districts = {
        n for names in json.loads((SOURCE / "districtsByCityCode.json").read_text("utf-8")).values() for n in names
    }
    units = json.load(gzip.open(SOURCE / "neighbourhoodsByDistrictAndCityCode.json.gz", "rt", encoding="utf-8"))

    extra = sorted(
        {
            line.strip()
            for line in EXTRA.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
    )
    single = {_key(n) for n in (*districts, *cities, *extra) if " " not in n}
    review = review_case_counts(args.reviews, single)
    places: dict[str, dict] = {}
    # Sonraki türler öncekini ezer: il > ilçe > ülke/bölge ("Akdeniz" hem ilçe hem bölge → ilçe)
    for kind, names in (("dis", extra), ("ilce", sorted(districts)), ("il", sorted(cities))):
        for name in names:
            if name in DROP:
                continue
            key = " ".join(_key(part) for part in name.split())
            stats = by_key[key]
            places[key] = {
                "name": name,
                "kind": kind,
                "tier": tier_for(name, stats, review[key]),
                # WikiNER'de yer adı olarak geçiş sayısı: küçük harfli çıplak eşleşme yalnız tanınmış adlarda
                "loc": stats[("LOCATION", "cap")] + stats[("LOCATION", "low")],
            }

    # "X Mahallesi" / "X köyü" kalıbı için tüm adlar. Mahalle adları tek başına eşleştirilmez:
    # ~21 bin tek sözcüklü ad arasında çok sayıda sözcük ve soyadı var ("Karaca", "Bektaşi", "Habibi").
    unit_names: set[str] = set()
    for city in units.values():
        for entries in city.values():
            for raw in entries:
                parsed = parse_unit(raw)
                if not parsed:
                    continue
                name, _kind = parsed
                unit_names.add(" ".join(_key(part) for part in name.split()))

    # Ek atılınca bir yer adına inen ama aslında yaygın sözcük olan küçük harfli biçimler
    # ("karşı" → ASCII "karsi" → "kars" + "ı"). Bunlar küçük harfli eşleşmede reddedilir.
    lookup = {key for key in places if " " not in key}
    collisions = sorted(
        word for word, count in lower_words.items() if word not in lookup and strip_suffix(word, lookup)
    )

    data = {
        "source": "PTT posta kodu listesi (turkey-neighbourhoods 4.0.3, MIT) + WikiNER eğitim istatistikleri",
        "places": places,
        "unit_names": sorted(unit_names),
        "lower_collisions": collisions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    args.output.write_bytes(gzip.compress(payload, mtime=0))  # mtime=0: aynı girdi → aynı dosya
    tiers = Counter(p["tier"] for p in places.values())
    print(
        json.dumps(
            {
                "il_ilce": len(places),
                "tiers": dict(tiers),
                "unit_names": len(unit_names),
                "lower_collisions": len(collisions),
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
