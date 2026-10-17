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
import time
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
        inference_timeout_seconds: float = 5.0,
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


# Eğitim ve değerlendirmedeki sözcük bölütlemesiyle aynı: "Market'in" tek sözcüktür (kesme işaretli ek dahil).
_WORD_TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)


class WordLevelPredictor:
    """Modeli eğitimdeki gibi sözcük düzeyinde çalıştırır: her sözcüğün etiketi ilk alt parçasından alınır.

    HF `pipeline` kesme işaretinde sözcüğü böler ("Market" + "'" + "in"); model eğitimde eklere hiç etiket
    görmediği için bu parçalara rastgele etiket verebilir. Burada sözcükler `is_split_into_words` ile verilir;
    böylece üretim, raporlanan değerlendirmeyle birebir aynı karar kuralını uygular. Uzun metin örtüşmeli
    sözcük pencerelerine bölünür; her sözcük, kenara en uzak olduğu pencereden etiket alır.
    """

    def __init__(self, tokenizer, model, device: int = -1, window_words: int = 100, overlap_words: int = 30):
        import torch

        if not 0 < overlap_words < window_words:
            raise ValueError("Pencere ayarı geçersiz.")
        self._torch = torch
        self.tokenizer = tokenizer
        self.device = torch.device("cpu" if device < 0 else f"cuda:{device}")
        self.model = model.to(self.device).eval()
        self.id2label = {int(k): v for k, v in model.config.id2label.items()}
        self.window_words = window_words
        self.overlap_words = overlap_words
        self.max_length = min(512, getattr(tokenizer, "model_max_length", 512) or 512)

    def _windows(self, count: int):
        step = self.window_words - self.overlap_words
        start = 0
        while True:
            end = min(count, start + self.window_words)
            core_start = start if start == 0 else start + self.overlap_words // 2
            core_end = end if end == count else end - self.overlap_words // 2
            yield start, end, core_start, core_end
            if end == count:
                return
            start += step

    def __call__(self, text: str) -> list[dict]:
        words = [(m.group(), m.start(), m.end()) for m in _WORD_TOKEN_RE.finditer(text)]
        if not words:
            return []
        labels = ["O"] * len(words)
        scores = [1.0] * len(words)
        torch = self._torch
        for start, end, core_start, core_end in self._windows(len(words)):
            encoded = self.tokenizer(
                [w for w, _, _ in words[start:end]],
                is_split_into_words=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            word_ids = encoded.word_ids(0)
            with torch.no_grad():
                logits = self.model(**{k: v.to(self.device) for k, v in encoded.items()}).logits[0]
            probs = torch.softmax(logits, dim=-1)
            seen: set[int] = set()
            for position, word_id in enumerate(word_ids):
                if word_id is None or word_id in seen:
                    continue
                seen.add(word_id)
                index = start + word_id
                if core_start <= index < core_end:
                    best = int(probs[position].argmax())
                    labels[index] = self.id2label.get(best, "O")
                    scores[index] = float(probs[position, best])
        results: list[dict] = []
        current = None
        for (_, w_start, w_end), label, score in zip(words, labels, scores, strict=True):
            kind = label[2:] if label[:2] in ("B-", "I-") else None
            if kind is None:
                current = None
                continue
            if label.startswith("I-") and current is not None and current["entity_group"] == kind:
                current["end"] = w_end
                current["_scores"].append(score)
                continue
            current = {"entity_group": kind, "start": w_start, "end": w_end, "_scores": [score]}
            results.append(current)
        for item in results:
            values = item.pop("_scores")
            item["score"] = sum(values) / len(values)
        return results


def _create_local_predictor(model_dir: str, device: int):
    """Load only local artifacts; used in the killable worker process."""
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("NER modeli için transformers kurulmalı.") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("Fast tokenizer gerekli.")
    model = AutoModelForTokenClassification.from_pretrained(model_dir, local_files_only=True)
    return WordLevelPredictor(tokenizer, model, device=device)


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
    """Killable local model worker with bounded startup and inference waits.

    Zaman aşımı ya da hata modeli kalıcı olarak kapatmaz: süreç sonlandırılır ve arka planda artan aralıklarla
    (``restart_delays``) yeniden başlatılır. Bu arada istekler kurallarla işlenir ve ``available`` False döner;
    API bunu yanıtında bildirir, eklenti kullanıcıyı uyarır.
    """

    def __init__(
        self,
        model_dir: str,
        *,
        device: int = -1,
        min_score: float = 0.0,
        startup_timeout_seconds: float = 60.0,
        inference_timeout_seconds: float = 5.0,
        restart_delays: tuple[float, ...] = (5.0, 15.0, 60.0, 300.0),
        keepalive_seconds: float = 60.0,
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
        self._closing = False
        self._restart_delays = tuple(restart_delays) or (5.0,)
        self._restart_thread: threading.Thread | None = None
        self.restarts = 0
        self.keepalive_seconds = keepalive_seconds
        self._last_used = time.monotonic()
        self._start_worker()
        atexit.register(self.close)
        if keepalive_seconds > 0:
            threading.Thread(target=self._keepalive_loop, name="ner-keepalive", daemon=True).start()

    @property
    def available(self) -> bool:
        return not self._disabled and not self._closing

    def _fail(self) -> None:
        """Süreci durdurur ve yeniden başlatmayı planlar (kilit altında çağrılır)."""
        self._disabled = True
        self._stop_worker()
        self._schedule_restart()

    # Boşta kalan model süreci (macOS bellek sıkıştırma / arka plan kısıtlaması) ilk mesajda yavaşlıyordu;
    # kısa bir deneme metniyle sıcak tutulur. Deneme de yanıt vermezse süreç, kullanıcı mesajından önce yenilenir.
    KEEPALIVE_TEXT = "Ali Ankara'da çalışıyor."

    def _keepalive_loop(self) -> None:
        while not self._closing:
            time.sleep(min(1.0, self.keepalive_seconds))
            if self._closing or self._disabled or time.monotonic() - self._last_used < self.keepalive_seconds:
                continue
            if not self._lock.acquire(blocking=False):
                continue  # gerçek istek işleniyor
            try:
                if self._disabled or self._connection is None:
                    continue
                self._connection.send(self.KEEPALIVE_TEXT)
                if not self._connection.poll(max(self.inference_timeout_seconds, 10.0)):
                    self._fail()
                    continue
                status, _payload = self._connection.recv()
                if status != "result":
                    self._fail()
                    continue
                self._last_used = time.monotonic()
            except (EOFError, OSError):
                self._fail()
            finally:
                self._lock.release()

    def _schedule_restart(self) -> None:
        if self._closing or (self._restart_thread is not None and self._restart_thread.is_alive()):
            return
        self._restart_thread = threading.Thread(target=self._restart_loop, name="ner-restart", daemon=True)
        self._restart_thread.start()

    def _restart_loop(self) -> None:
        attempt = 0
        while not self._closing:
            delay = self._restart_delays[min(attempt, len(self._restart_delays) - 1)]
            deadline = time.monotonic() + delay
            while not self._closing and time.monotonic() < deadline:
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            if self._closing:
                return
            try:
                # Yükleme kilit dışında yapılır: bu sırada istekler kurallarla işlenmeye devam eder
                process, connection = self._spawn_worker()
            except BaseException:
                attempt += 1
                continue
            with self._lock:
                if self._closing:
                    self._process, self._connection = process, connection
                    self._stop_worker()
                    return
                self._process, self._connection = process, connection
                self._disabled = False
                self.restarts += 1
            logger.warning("NER modeli yeniden başlatıldı.")
            return

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

    def _spawn_worker(self):
        """Yeni model süreci başlatır ve hazır olmasını bekler; başarısızsa süreci kapatıp hata verir."""
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_ner_process_main, args=(child, self.model_dir, self.device), daemon=True
        )
        try:
            process.start()
            child.close()
            if not parent.poll(self.startup_timeout_seconds):
                raise TimeoutError("NER model yükleme zaman aşımı.")
            status, _payload = parent.recv()
            if status != "ready":
                raise RuntimeError("NER model yüklenemedi.")
        except BaseException:
            for handle in (parent, child):
                try:
                    handle.close()
                except OSError:
                    pass
            if process.is_alive():
                process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
            raise
        return process, parent

    def _start_worker(self) -> None:
        try:
            self._process, self._connection = self._spawn_worker()
        except BaseException:
            self._disabled = True
            raise

    def recognize(self, text: str) -> Iterable[EntitySpan]:
        if self._disabled:
            raise RuntimeError("NER worker yeniden başlatılıyor; bu istek kural tabanlı işlenecek.")
        if not self._lock.acquire(timeout=self.inference_timeout_seconds):
            raise TimeoutError("NER worker meşgul; bu istek kural tabanlı işlenecek.")
        try:
            if self._disabled or self._connection is None or self._process is None or not self._process.is_alive():
                self._fail()
                raise RuntimeError("NER worker kullanılamıyor.")
            try:
                self._last_used = time.monotonic()
                self._connection.send(text)
                if not self._connection.poll(self.inference_timeout_seconds):
                    self._fail()
                    raise TimeoutError("NER inference zaman aşımı.")
                status, payload = self._connection.recv()
            except (EOFError, OSError) as exc:
                self._fail()
                raise RuntimeError("NER worker yanıt vermedi.") from exc
            if status != "result":
                self._fail()
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
        self._closing = True
        if self._lock.acquire(timeout=self.inference_timeout_seconds):
            try:
                self._disabled = True
                self._stop_worker()
            finally:
                self._lock.release()


_MERGE_GAP = frozenset(" \t,/-–")  # ";" ve satır sonu ayrı adresleri ayırır


_TRAILING_SUFFIX_RE = re.compile(r"['’][^\W\d_]+$", re.UNICODE)


def strip_apostrophe_suffix(text: str, span: EntitySpan) -> EntitySpan:
    """Kesme işaretinden sonraki ek varlığa ait değildir: "Ayşe Demir'e" → "Ayşe Demir" (anotasyon kuralı).

    Sözcük düzeyindeki model eki de içeren sözcüğün tamamını işaretler; ek dışarıda bırakılınca hem aynı kişi
    farklı eklerle aynı yer tutucuyu alır hem de ek LLM'e gider ("[KİŞİ_1]'e söyle").
    """
    match = _TRAILING_SUFFIX_RE.search(text[span.start : span.end])
    if match and match.start() > 0:
        return EntitySpan(span.type, span.start, span.start + match.start())
    return span


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


_DETACHED_SUFFIX_RE = re.compile(r"\s+(?:a|e|ya|ye|da|de|ta|te|dan|den|tan|ten|daki|deki)$", re.IGNORECASE)


def _union_addresses(
    text: str, authoritative: list[Detection], candidates: list[Detection], rules: list[Detection]
) -> tuple[list[Detection], list[Detection]]:
    """Öncelikli ADDRESS span'ı, çakıştığı aday ADDRESS span'larıyla birleşir (en geniş kapsam).

    Birleşim deterministik kural tespitlerine (TCKN, IBAN, telefon...), ";" ya da satır sonuna taşmaz.
    """
    model_addresses = [c for c in candidates if c.type == "ADDRESS"]
    if not model_addresses:
        return authoritative, candidates
    absorbed: set[int] = set()
    result: list[Detection] = []
    for item in authoritative:
        if item.type == "ADDRESS":
            low = max((r.end for r in rules if r.end <= item.start), default=0)
            previous_separator = max(text.rfind(";", 0, item.start), text.rfind("\n", 0, item.start))
            low = max(low, previous_separator + 1)
            high = min((r.start for r in rules if r.start >= item.end), default=len(text))
            # ";" ve satır sonu ayrı kayıtları ayırır (bkz. merge_adjacent_spans)
            separator = re.search(r"[;\n]", text[item.end :])
            if separator:
                high = min(high, item.end + separator.start())
            start, end = item.start, item.end
            grew = True
            while grew:
                grew = False
                for other in model_addresses:
                    if id(other) not in absorbed and other.start < end and other.end > start:
                        start, end = max(low, min(start, other.start)), min(high, max(end, other.end))
                        absorbed.add(id(other))
                        grew = True
            # Ayrı yazılmış hâl eki adrese ait değildir: "... trabzon a bırak" → "... trabzon"
            detached = _DETACHED_SUFFIX_RE.search(text[start:end])
            if detached and detached.start() > 0:
                end = start + detached.start()
            item = Detection("ADDRESS", start, end, text[start:end])
        result.append(item)
    kept = [c for c in candidates if id(c) not in absorbed]
    return result, kept


# Numarasız tesis/bina adının son sözcüğü ("Zümrüt Apartmanı", "Çınar Plaza", "Akasya İş Merkezi")
FACILITY_HEAD_RE = re.compile(
    r"(apartman\w*|apt|plaza\w*|park[ıi]\w*|kampüs\w*|kampus\w*|pasaj\w*|rezidans\w*|avm"
    r"|(?:iş|is|alışveriş|alisveris) (?:merkez\w*|han[ıi]\w*))$",
    re.IGNORECASE,
)


def merge_facility_into_address(text: str, detections: list[Detection]) -> list[Detection]:
    """Bina/tesis adı (LOC) hemen ardından gelen açık adresle tek ADDRESS olur.

    Anotasyon kuralı: numaralı adresin önündeki bina bilgisi adresin parçasıdır
    ("Gül Apartmanı, Kiraz Sokak No 21, Kayseri"). Arada yalnız boşluk / virgül olabilir.
    """
    items = sorted(detections, key=lambda d: d.start)
    out: list[Detection] = []
    index = 0
    while index < len(items):
        item = items[index]
        nxt = items[index + 1] if index + 1 < len(items) else None
        if (
            nxt is not None
            and item.type == "LOCATION"
            and nxt.type == "ADDRESS"
            and FACILITY_HEAD_RE.search(re.split(r"['’]", item.value)[0].strip())
            and text[item.end : nxt.start].strip(" ,") == ""
            and "\n" not in text[item.end : nxt.start]
        ):
            out.append(Detection("ADDRESS", item.start, nxt.end, text[item.start : nxt.end]))
            index += 2
            continue
        out.append(item)
        index += 1
    return out


class DetectionPipeline:
    """Combine deterministic rule matches with optional NER recognizers."""

    def __init__(self, recognizers: Iterable[EntityRecognizer] = (), *, address_union: bool = True):
        self.recognizers = tuple(recognizers)
        # Öncelikli (kural) adres ile çakışan model adresi birleşimle (en geniş kapsam) çözülür;
        # DLP'de eksik maskeleme fazladan maskelemeden pahalıdır. False: v1 davranışı (kural span'ı aynen).
        self.address_union = address_union
        self._failure_logged: set[int] = set()

    def detect(self, text: str) -> list[Detection]:
        return self.detect_with_status(text)[0]

    def detect_with_status(self, text: str) -> tuple[list[Detection], bool]:
        """Tespitler ve bu istekte bir tanıyıcının başarısız olup olmadığı (True: yalnız kurallarla işlendi)."""
        degraded = False
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
                degraded = True
                if recognizer_index not in self._failure_logged:
                    logger.warning("NER tanıyıcısı başarısız; kural tabanlı tespit ile devam ediliyor.")
                    self._failure_logged.add(recognizer_index)
                continue
            self._failure_logged.discard(recognizer_index)  # düzelince bir sonraki arıza yine günlüğe yazılır
            for span in spans:
                if not (0 <= span.start < span.end <= len(text)):
                    raise ValueError("NER tanıyıcısı metin sınırları dışında offset döndürdü.")
            normalized = [
                EntitySpan(ENTITY_ALIASES.get(span.type.upper(), span.type), span.start, span.end) for span in spans
            ]
            # Önce kırp (";" gibi ayırıcılar span dışına çıksın), sonra birleştir, sonra tekrar kırp.
            trimmed = [trim_address_span(text, strip_apostrophe_suffix(text, item)) for item in normalized]
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

        if self.address_union:
            authoritative, candidates = _union_addresses(text, authoritative, candidates, rules)
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
        if self.address_union:
            merged = merge_facility_into_address(text, merged)
        return sorted([*rules, *_resolve_merged(text, merged, ner_items, rules)], key=lambda d: d.start), degraded


def _overlaps(a: Detection, b: Detection) -> bool:
    return a.start < b.end and a.end > b.start


def _resolve_merged(
    text: str, merged: list[Detection], originals: list[Detection], rules: list[Detection]
) -> list[Detection]:
    """Birleştirme sonrası span'ları maskelenebilir kılar: kurallarla ve birbirleriyle çakışmasınlar.

    Kural tespitinin (TCKN/IBAN...) üstünden uzayan birleşik span atılır ama içindeki özgün parçalar korunur
    (tümden atmak, model bulduğu hâlde kurum adını maskesiz bırakırdı). Birbiriyle çakışan span'lar tek span'da
    birleşir; tür uzun olanınkidir. Eksik maskeleme fazladan maskelemeden pahalıdır.
    """
    safe: list[Detection] = []
    for item in merged:
        if any(_overlaps(item, rule) for rule in rules):
            safe.extend(o for o in originals if item.start <= o.start and o.end <= item.end)
        else:
            safe.append(item)
    resolved: list[Detection] = []
    for item in sorted(safe, key=lambda d: (d.start, -(d.end - d.start))):
        previous = resolved[-1] if resolved else None
        if previous is None or not _overlaps(previous, item):
            resolved.append(item)
            continue
        if item.end <= previous.end:
            continue  # tamamen içeride
        kind = previous.type if previous.end - previous.start >= item.end - item.start else item.type
        resolved[-1] = Detection(kind, previous.start, item.end, text[previous.start : item.end])
    return resolved
