"""Optional adapter seam for a future Turkish NER model.

The core deliberately ships with no model dependency or downloaded weights.
Recognizers must return offsets into the original input string.
"""

from __future__ import annotations

import atexit
import logging
import math
import multiprocessing
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .detectors import Detection, detect

NER_TYPES = frozenset({"PERSON", "ORG", "LOCATION", "ADDRESS"})
ENTITY_ALIASES = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "ORG": "ORG",
    "LOC": "LOCATION",
    "GPE": "LOCATION",
    "LOCATION": "LOCATION",
    "ADDRESS": "ADDRESS",
}
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EntitySpan:
    type: str
    start: int
    end: int


class EntityRecognizer(Protocol):
    def recognize(self, text: str) -> Iterable[EntitySpan]:
        """Return entity spans using Python string offsets in ``text``."""


class HuggingFaceNERRecognizer:
    """Adapter for a locally saved Hugging Face token-classification model.

    Pass an injected callable for tests, or use ``from_local_model`` after a
    trained model directory has been produced. Model loading never happens at
    module import time and local-only loading avoids surprise Hub downloads.
    """

    def __init__(self, predictor, min_score: float = 0.0):
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score 0 ile 1 arasında olmalı.")
        self.predictor = predictor
        self.min_score = min_score

    @classmethod
    def from_local_model(
        cls,
        model_dir: str,
        *,
        device: int = -1,
        min_score: float = 0.0,
        startup_timeout_seconds: float = 60.0,
        inference_timeout_seconds: float = 2.0,
    ):
        return ProcessIsolatedHuggingFaceNERRecognizer(
            model_dir,
            device=device,
            min_score=min_score,
            startup_timeout_seconds=startup_timeout_seconds,
            inference_timeout_seconds=inference_timeout_seconds,
        )

    def recognize(self, text: str) -> Iterable[EntitySpan]:
        predictions = self.predictor(text)
        spans = []
        for prediction in predictions:
            label = prediction.get("entity_group", prediction.get("entity", ""))
            # Some pipelines emit BIO labels while others emit entity groups.
            if label.startswith(("B-", "I-")):
                label = label[2:]
            kind = ENTITY_ALIASES.get(label.upper())
            if kind not in NER_TYPES or float(prediction.get("score", 1.0)) < self.min_score:
                continue
            start, end = prediction.get("start"), prediction.get("end")
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError("NER modeli karakter offset'i döndürmedi.")
            spans.append(EntitySpan(kind, start, end))
        return spans


def _create_local_predictor(model_dir: str, device: int):
    """Load only local artifacts; used in the killable worker process."""
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
    except ImportError as exc:
        raise RuntimeError("NER modeli için transformers kurulmalı.") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("Fast tokenizer gerekli.")
    model = AutoModelForTokenClassification.from_pretrained(model_dir, local_files_only=True)
    return pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="first",  # eğitimdeki ilk alt sözcük etiketlemesiyle aynı; sözcük ortadan bölünmez
        stride=64,
        device=device,
    )


def _ner_process_main(connection, model_dir: str, device: int) -> None:
    try:
        predictor = _create_local_predictor(model_dir, device)
        connection.send(("ready", None))
    except BaseException:
        try:
            connection.send(("error", None))
        except (BrokenPipeError, EOFError, OSError):
            pass
        connection.close()
        return

    while True:
        try:
            text = connection.recv()
        except (EOFError, OSError):
            break
        if text is None:
            break
        try:
            rows = predictor(text)
            # Limit IPC response to model labels, scores, and original offsets.
            predictions = [
                {key: row[key] for key in ("entity_group", "entity", "start", "end", "score") if key in row}
                for row in rows
            ]
            connection.send(("result", predictions))
        except BaseException:
            # Do not send exception messages or input text to the parent/logs.
            try:
                connection.send(("error", None))
            except (BrokenPipeError, EOFError, OSError):
                break
    connection.close()


class ProcessIsolatedHuggingFaceNERRecognizer:
    """Killable local model worker with bounded startup and inference waits."""

    def __init__(
        self,
        model_dir: str,
        *,
        device: int = -1,
        min_score: float = 0.0,
        startup_timeout_seconds: float = 60.0,
        inference_timeout_seconds: float = 2.0,
    ):
        if not Path(model_dir).is_dir():
            raise FileNotFoundError("Yerel NER model klasörü bulunamadı.")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score 0 ile 1 arasında olmalı.")
        if (
            not math.isfinite(startup_timeout_seconds)
            or not math.isfinite(inference_timeout_seconds)
            or not 0 < startup_timeout_seconds <= 600
            or not 0 < inference_timeout_seconds <= 60
        ):
            raise ValueError("NER yükleme timeout'u 0-600, inference timeout'u 0-60 saniye aralığında olmalı.")
        self.model_dir = model_dir
        self.device = device
        self.min_score = min_score
        self.startup_timeout_seconds = startup_timeout_seconds
        self.inference_timeout_seconds = inference_timeout_seconds
        self._context = multiprocessing.get_context("spawn")
        self._lock = threading.Lock()
        self._process = None
        self._connection = None
        self._disabled = False
        self._start_worker()
        atexit.register(self.close)

    def _stop_worker(self) -> None:
        connection, process = self._connection, self._process
        self._connection = self._process = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
            if not process.is_alive() and hasattr(process, "close"):
                process.close()

    def _start_worker(self) -> None:
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_ner_process_main, args=(child, self.model_dir, self.device), daemon=True
        )
        try:
            process.start()
            self._process, self._connection = process, parent
            child.close()
            if not parent.poll(self.startup_timeout_seconds):
                raise TimeoutError("NER model yükleme zaman aşımı.")
            status, _payload = parent.recv()
            if status != "ready":
                raise RuntimeError("NER model yüklenemedi.")
        except BaseException:
            self._disabled = True
            try:
                parent.close()
            except OSError:
                pass
            try:
                child.close()
            except OSError:
                pass
            self._stop_worker()
            raise

    def recognize(self, text: str) -> Iterable[EntitySpan]:
        if self._disabled:
            raise RuntimeError("NER worker timeout veya yükleme hatası nedeniyle devre dışı.")
        if not self._lock.acquire(timeout=self.inference_timeout_seconds):
            raise TimeoutError("NER worker meşgul; bu istek kural tabanlı işlenecek.")
        try:
            if self._disabled or self._connection is None or self._process is None or not self._process.is_alive():
                self._disabled = True
                self._stop_worker()
                raise RuntimeError("NER worker kullanılamıyor.")
            try:
                self._connection.send(text)
                if not self._connection.poll(self.inference_timeout_seconds):
                    self._disabled = True
                    self._stop_worker()
                    raise TimeoutError("NER inference zaman aşımı.")
                status, payload = self._connection.recv()
            except (EOFError, OSError) as exc:
                self._disabled = True
                self._stop_worker()
                raise RuntimeError("NER worker yanıt vermedi.") from exc
            if status != "result":
                self._disabled = True
                self._stop_worker()
                raise RuntimeError("NER inference hatası.")
            spans = []
            for prediction in payload:
                label = prediction.get("entity_group", prediction.get("entity", ""))
                if label.startswith(("B-", "I-")):
                    label = label[2:]
                kind = ENTITY_ALIASES.get(label.upper())
                if kind not in NER_TYPES or float(prediction.get("score", 1.0)) < self.min_score:
                    continue
                start, end = prediction.get("start"), prediction.get("end")
                if not isinstance(start, int) or not isinstance(end, int):
                    raise ValueError("NER modeli karakter offset'i döndürmedi.")
                spans.append(EntitySpan(kind, start, end))
            return spans
        finally:
            self._lock.release()

    def close(self) -> None:
        if self._lock.acquire(timeout=self.inference_timeout_seconds):
            try:
                self._disabled = True
                self._stop_worker()
            finally:
                self._lock.release()


_MERGE_GAP = frozenset(" \t,/-–")  # ";" ve satır sonu ayrı adresleri ayırır


def merge_adjacent_spans(text: str, spans: Iterable[EntitySpan]) -> list[EntitySpan]:
    """Bitişik aynı türden ADDRESS parçalarını tek span'a birleştirir.

    Model bir adresi noktalamada bölebilir ("…No: 5" + ", X Mahallesi"). Aradaki
    metin yalnızca boşluk/virgül/eğik çizgi/tire ise iki parça tek adres sayılır;
    böylece tek placeholder üretilir ve adresin bağlamı LLM'e parça parça gitmez.
    """
    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    merged: list[EntitySpan] = []
    for span in ordered:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and previous.type == span.type == "ADDRESS"
            and previous.end <= span.start
            and all(char in _MERGE_GAP for char in text[previous.end : span.start])
        ):
            merged[-1] = EntitySpan("ADDRESS", previous.start, max(previous.end, span.end))
        else:
            merged.append(span)
    return merged


# Adres sonunda noktası kendine ait olan kısaltmalar (küçük harf, noktasız).
ADDRESS_ABBREVIATIONS = frozenset(
    {"mah", "mh", "sk", "sok", "cad", "cd", "bulv", "blv", "no", "d", "k", "apt", "ap", "bl", "sit", "ltd", "şti", "aş"}
)
_TRAILING = ".,;:"


def trim_address_span(text: str, span: EntitySpan) -> EntitySpan:
    """Adres sonundaki cümle noktalamasını span dışına alır ("…Mahallesi." → "…Mahallesi").

    Nokta bir kısaltmaya aitse ("Sk.", "Mah.", "No.") korunur. Model cümle sonu
    noktasını adrese katmaya eğilimli olduğundan bu, sınırları tutarlı kılar.
    """
    if span.type != "ADDRESS":
        return span
    start, end = span.start, span.end
    while end > start and text[end - 1] in _TRAILING + " ":
        if text[end - 1] == ".":
            word = text[start : end - 1].rsplit(maxsplit=1)[-1] if text[start : end - 1].strip() else ""
            if word.casefold().translate(str.maketrans("", "", ".")) in ADDRESS_ABBREVIATIONS or word.endswith("."):
                break
        end -= 1
    return EntitySpan(span.type, start, end) if end > start else span


def plausible_address(value: str) -> bool:
    """Anotasyon kuralı: açık adres numara (kapı/daire/kat) ve en az bir sözcük içerir.

    Modelin ürettiği ": 5" ya da "blok" gibi parçalar bu koşulu sağlamaz ve elenir.
    """
    has_digit = any(char.isdigit() for char in value)
    words = [w for w in value.replace("/", " ").split() if sum(c.isalpha() for c in w) >= 2]
    return has_digit and len(words) >= 1 and len(value.strip()) >= 6


_NUMBER_WORDS = frozenset({"no", "numara", "nu", "kat", "daire"})


def has_name_word(value: str) -> bool:
    """Kişi/kurum/yer span'ı en az bir gerçek sözcük içermeli ("No : 5" gibi parçalar elenir)."""
    for word in value.split():
        letters = "".join(char for char in word if char.isalpha())
        digits = sum(char.isdigit() for char in word)
        if letters.casefold() in _NUMBER_WORDS:
            continue
        if len(letters) >= 2 or (letters and digits >= 2):  # "A101" gibi marka kodları
            return True
    return False


_TR_FOLD = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
# Kurumsal baş sözcükler ve alt birim sözcükleri (anotasyon kuralı: alt birim dahil tek ORG).
INSTITUTION_HEADS = frozenset(
    "holding bankasi universitesi hastanesi belediyesi bakanligi mudurlugu baskanligi kurumu kurulu dernegi "
    "vakfi odasi borsasi mahkemesi savciligi valiligi kaymakamligi lisesi koleji fakultesi enstitusu sirketi "
    "kooperatifi federasyonu sendikasi ajansi dairesi merkezi akademisi poliklinigi klinigi bolumu muhendisligi "
    "subesi birimi servisi dekanligi rektorlugu laboratuvari adliyesi".split()
)
_WORD_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]")


def _head_key(word: str) -> str:
    base = re.split(r"['’]", word, maxsplit=1)[0]
    return base.replace("İ", "i").replace("I", "ı").lower().translate(_TR_FOLD)


def _ends_with_head(value: str) -> bool:
    words = value.split()
    return bool(words) and _head_key(words[-1]) in INSTITUTION_HEADS


def merge_org_compounds(text: str, detections: list[Detection]) -> list[Detection]:
    """Kurum adlarını anotasyon kuralına göre bütünler.

    1. Şehir + kurum: "Antalya" (LOC) + "Kurumlar Vergi Dairesi" (ORG) → tek ORG.
    2. Kurum + alt birim: ORG'dan sonra büyük harfle yazılmış ve kurumsal bir sözcükle biten
       ifade ("… Üniversitesi" + "Makine Fakültesi"), arada yer adı olsa bile → tek ORG.
    Yalnızca boşlukla ayrılmış bitişik parçalar birleşir; başka türdeki tespitler korunur.
    """
    items = sorted(detections, key=lambda d: d.start)
    changed = True
    while changed:
        changed = False
        for i, item in enumerate(items):
            if item.type != "ORG":
                continue
            # (1) önceki LOC'u yut
            prev_loc = items[i - 1] if i > 0 else None
            if (
                prev_loc is not None
                and prev_loc.type == "LOCATION"
                and not re.search(r"['’]", prev_loc.value)  # "Ankara'da Ankara Ticaret Odası" birleşmez
                and text[prev_loc.end : item.start] == " "
            ):
                if _ends_with_head(item.value):
                    prev = items[i - 1]
                    items[i - 1 : i + 1] = [Detection("ORG", prev.start, item.end, text[prev.start : item.end])]
                    changed = True
                    break
            # (2) ardından gelen alt birimi ekle
            tail = text[item.end :]
            if tail.startswith(("'", "’")):
                continue
            cursor = item.end
            last_head_end = None
            for match in _WORD_RE.finditer(tail):
                word = match.group()
                start = item.end + match.start()
                gap = text[cursor:start]
                if gap.strip() or "\n" in gap:
                    break
                if not (word[:1].isupper() or word.isdigit() or word == "."):
                    break
                cursor = start + len(re.split(r"['’]", word, maxsplit=1)[0])
                if _head_key(word) in INSTITUTION_HEADS:
                    last_head_end = cursor
                if "'" in word or "’" in word:
                    break
            if last_head_end is not None:
                covered = [d for d in items if d is not item and d.start >= item.end and d.end <= last_head_end]
                if all(d.type in ("LOCATION", "ORG") for d in covered):
                    merged = Detection("ORG", item.start, last_head_end, text[item.start : last_head_end])
                    items = [d for d in items if d is not item and d not in covered] + [merged]
                    items.sort(key=lambda d: d.start)
                    changed = True
                    break
    return items


class DetectionPipeline:
    """Combine deterministic rule matches with optional NER recognizers."""

    def __init__(self, recognizers: Iterable[EntityRecognizer] = ()):
        self.recognizers = tuple(recognizers)
        self._failure_logged: set[int] = set()

    def detect(self, text: str) -> list[Detection]:
        rules = detect(text)
        selected = list(rules)
        candidates: list[Detection] = []
        authoritative: list[Detection] = []
        fallback: list[Detection] = []
        for recognizer_index, recognizer in enumerate(self.recognizers):
            # Bir tanıyıcı bazı türlerde sınırları daha güvenilir bulabilir (ör. kural katmanı ADDRESS);
            # o türdeki span'ları çakışan diğer tanıyıcı span'larının önüne geçer.
            priority_types = getattr(recognizer, "authoritative_types", frozenset())
            # Bazı türlerde ise yalnızca diğer tanıyıcıların boş bıraktığı yeri doldurur (ör. kural ORG).
            fallback_types = getattr(recognizer, "fallback_types", frozenset())
            try:
                spans = tuple(recognizer.recognize(text))
            except Exception:
                # Project policy: a model runtime failure degrades to
                # deterministic detectors. Never log the input or exception,
                # which may contain request data.
                if recognizer_index not in self._failure_logged:
                    logger.warning("NER tanıyıcısı başarısız; kural tabanlı tespit ile devam ediliyor.")
                    self._failure_logged.add(recognizer_index)
                continue
            for span in spans:
                if not (0 <= span.start < span.end <= len(text)):
                    raise ValueError("NER tanıyıcısı metin sınırları dışında offset döndürdü.")
            normalized = [
                EntitySpan(ENTITY_ALIASES.get(span.type.upper(), span.type), span.start, span.end) for span in spans
            ]
            # Önce kırp (";" gibi ayırıcılar span dışına çıksın), sonra birleştir, sonra tekrar kırp.
            trimmed = [trim_address_span(text, item) for item in normalized]
            for span in (trim_address_span(text, item) for item in merge_adjacent_spans(text, trimmed)):
                kind = ENTITY_ALIASES.get(span.type.upper())
                if kind not in NER_TYPES:
                    raise ValueError(f"NER tanıyıcısı desteklenmeyen tür döndürdü: {span.type}")
                if not (0 <= span.start < span.end <= len(text)):
                    raise ValueError("NER tanıyıcısı metin sınırları dışında offset döndürdü.")
                item = Detection(kind, span.start, span.end, text[span.start : span.end])
                if kind == "ADDRESS" and not plausible_address(item.value):
                    continue
                if kind != "ADDRESS" and not has_name_word(item.value):
                    continue
                if kind in priority_types:
                    authoritative.append(item)
                elif kind in fallback_types:
                    fallback.append(item)
                else:
                    candidates.append(item)

        # Deterministic rules take precedence over model spans. Model spans
        # that overlap each other are resolved longest-first at each position.
        ordered = sorted(authoritative, key=lambda d: (d.start, -(d.end - d.start), d.type))
        ordered += sorted(candidates, key=lambda d: (d.start, -(d.end - d.start), d.type))
        ordered += sorted(fallback, key=lambda d: (d.start, -(d.end - d.start), d.type))
        for item in ordered:
            if not any(item.start < prior.end and item.end > prior.start for prior in selected):
                selected.append(item)
        ner_items = [d for d in selected if d not in rules]
        merged = merge_org_compounds(text, ner_items)
        # Birleştirme deterministik kural tespitleriyle (TCKN/IBAN...) çakışamaz.
        merged = [d for d in merged if not any(d.start < r.end and d.end > r.start for r in rules)]
        return sorted([*rules, *merged], key=lambda d: d.start)
