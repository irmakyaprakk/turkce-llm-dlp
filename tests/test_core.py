import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from training.ner.evaluate_bypass import evaluate
from training.ner.prepare_data import prepare, synthetic_address_rows
from training.ner.train import read_jsonl
from turkce_llm_dlp import api as api_module
from turkce_llm_dlp.api import app
from turkce_llm_dlp.detectors import detect, normalized_view, shannon_entropy, valid_card_number, valid_iban, valid_tckn
from turkce_llm_dlp.gateway import Gateway, OpenAICompatibleProvider, UpstreamUnavailable
from turkce_llm_dlp.masking import MaskingSession, process_demo
from turkce_llm_dlp.ner import (
    DetectionPipeline,
    EntitySpan,
    HuggingFaceNERRecognizer,
    ProcessIsolatedHuggingFaceNERRecognizer,
    _create_local_predictor,
)
from turkce_llm_dlp.policy import Policy


def test_checksums():
    assert valid_tckn("10000000146")
    assert not valid_tckn("10000000147")
    assert valid_iban("TR330006100519786457841326")
    assert not valid_iban("TR330006100519786457841327")
    assert valid_card_number("4111 1111 1111 1111")
    assert not valid_card_number("4111 1111 1111 1112")


def test_luhn_card_is_masked_consistently_and_invalid_candidate_is_ignored():
    text = "Kart 4111-1111-1111-1111, tekrar 4111 1111 1111 1111; hatalı 4111 1111 1111 1112"
    session = MaskingSession()
    masked, detections = session.mask(text)
    assert masked.count("<CARD_1>") == 2
    assert "4111 1111 1111 1112" in masked
    assert detections[0]["type"] == "CARD"


def test_bypass_evaluation_uses_minimum_sample_size_and_reports_both_conditions():
    report = evaluate(samples_per_class=50, seed=7)
    assert set(report["techniques"]) == {"separators", "zero_width", "fullwidth"}
    assert report["techniques"]["zero_width"]["normalization_on"]["capture_rate"] == 1.0
    assert report["techniques"]["fullwidth"]["normalization_on"]["false_positive_rate"] == 0.0


def test_normalization_preserves_original_offsets():
    original = "TCKN: １０００００００１４６"
    view, offsets = normalized_view(original)
    assert "10000000146" in view
    item = next(d for d in detect(original) if d.type == "TCKN")
    assert original[item.start : item.end] == "１０００００００１４６"
    assert offsets


def test_separators_zero_width_and_duplicate_consistency():
    text = "10000\u200b000146 / 10000-000146"
    session = MaskingSession()
    masked, found = session.mask(text)
    assert masked == "<TCKN_1> / <TCKN_1>"
    assert len(found) == 2
    assert session.restore("Kimlik <TCKN_1>") == "Kimlik 10000\u200b000146"


def test_email_and_known_secret_detection():
    matches = detect("iletisim: kisi@example.com. token=ghp_abcdefghijk123456")
    assert [item.type for item in matches] == ["EMAIL", "SECRET"]


def test_demo_response_restores_only_current_request_values():
    result = process_demo("Merhaba 10000000146")
    assert "<TCKN_1>" in result["masked_prompt"]
    assert "10000000146" not in result["mock_response"]
    assert "10000000146" in result["restored_response"]


def test_placeholder_collision_and_stream_boundaries():
    session = MaskingSession()
    masked, _ = session.mask("Literal <TCKN_1>, actual 10000000146")
    assert masked == "Literal <TCKN_1>, actual <TCKN_2>"
    chunks = ["Cevap: <TCK", "N_2", "> tamam"]
    assert "".join(session.restore_chunks(chunks)) == "Cevap: 10000000146 tamam"


def test_api_limits_and_does_not_return_detected_values():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok", "ner": "off"}  # model yapılandırılmamış
    response = client.post("/v1/detect", json={"text": "TCKN 10000000146"})
    assert response.status_code == 200
    assert "10000000146" not in response.text
    assert response.json()["detections"][0]["type"] == "TCKN"
    assert response.json()["detections"][0]["action"] == "mask"

    too_large = client.post("/v1/process", content=json.dumps({"text": "x" * 100_000}))
    assert too_large.status_code == 413


def test_yaml_policy_and_block_prevents_upstream_call():
    policy = Policy.from_yaml(Path(__file__).parents[1] / "config" / "policy.example.yml")

    class NeverProvider:
        called = False

        async def complete(self, prompt):
            self.called = True
            raise AssertionError("blocked prompt reached provider")

    provider = NeverProvider()
    result = asyncio.run(Gateway(provider, policy).process("TCKN 10000000146"))
    assert result["blocked"] is True
    assert result["detections"][0]["action"] == "block"
    assert provider.called is False
    assert Policy(default_action="allow").action_for("FUTURE_ENTITY") == "mask"


def test_gateway_provider_receives_only_masked_text_and_restores_response():
    class CapturingProvider:
        prompt = None

        async def complete(self, prompt):
            self.prompt = prompt
            return "Yanıt: " + prompt

    provider = CapturingProvider()
    result = asyncio.run(Gateway(provider).process("Kullanıcı 10000000146"))
    assert "10000000146" not in provider.prompt
    assert "<TCKN_1>" in provider.prompt
    assert "10000000146" in result["response"]


def test_gateway_provider_failure_is_sanitized():
    class BrokenProvider:
        async def complete(self, prompt):
            raise RuntimeError("sensitive request data 10000000146")

    try:
        asyncio.run(Gateway(BrokenProvider()).process("10000000146"))
    except UpstreamUnavailable as exc:
        assert "10000000146" not in str(exc)
    else:
        raise AssertionError("expected UpstreamUnavailable")


def test_gateway_endpoint_uses_demo_provider():
    response = TestClient(app).post("/v1/gateway", json={"text": "Iletişim kisi@example.com"})
    assert response.status_code == 200
    assert response.json()["masked_count"] == 1
    assert "kisi@example.com" in response.json()["response"]


def test_openai_compatible_adapter_sends_masked_prompt_only():
    captured = {}

    def handler(request):
        captured["authorization"] = request.headers["authorization"]
        captured["payload"] = json.loads(request.content)
        prompt = captured["payload"]["messages"][0]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "Yanıt: " + prompt}}]})

    provider = OpenAICompatibleProvider(
        api_key="test-secret",
        model="test-model",
        endpoint="https://mock.example/v1/chat/completions",
        transport=httpx.MockTransport(handler),
    )
    result = asyncio.run(Gateway(provider).process("TCKN 10000000146"))
    assert captured["authorization"] == "Bearer test-secret"
    assert "10000000146" not in captured["payload"]["messages"][0]["content"]
    assert result["response"].endswith("10000000146")


def test_provider_requires_tls_except_localhost():
    try:
        OpenAICompatibleProvider(api_key="x", model="m", endpoint="http://public.example/api")
    except ValueError as exc:
        assert "HTTPS" in str(exc)
    else:
        raise AssertionError("plain HTTP endpoint should be rejected")


def test_streaming_gateway_restores_placeholder_split_across_many_fragments():
    response = TestClient(app).post("/v1/gateway/stream", json={"text": "TCKN 10000000146"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-store"
    assert '"delta": "10000000146"' in response.text
    assert "[DONE]" in response.text


def test_openai_compatible_sse_parser_handles_fragmented_wire_chunks():
    class SplitStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for chunk in [b'data: {"choices":[{"delta":{"content":"<TCK', b'N_1>"}}]}\r\n\r\n', b"data: [DONE]\n"]:
                yield chunk

    def handler(_request):
        return httpx.Response(200, stream=SplitStream())

    provider = OpenAICompatibleProvider(
        api_key="test-secret",
        model="test-model",
        endpoint="https://mock.example/v1/chat/completions",
        transport=httpx.MockTransport(handler),
    )

    async def collect():
        return [part async for part in provider.stream("safe prompt")]

    assert asyncio.run(collect()) == ["<TCKN_1>"]


def test_optional_ner_pipeline_masks_person_and_keeps_rule_detector_priority():
    class FakeTurkishRecognizer:
        def recognize(self, text):
            person_start = text.index("Ayşe Yılmaz")
            email_start = text.index("ayse@example.com")
            return [
                EntitySpan("PER", person_start, person_start + len("Ayşe Yılmaz")),
                # A bad overlapping model span must not replace a deterministic email rule.
                EntitySpan("ORG", email_start, email_start + len("ayse@example.com")),
            ]

    pipeline = DetectionPipeline([FakeTurkishRecognizer()])
    result = pipeline.detect("Ayşe Yılmaz ayse@example.com")
    assert [(item.type, item.value) for item in result] == [
        ("PERSON", "Ayşe Yılmaz"),
        ("EMAIL", "ayse@example.com"),
    ]


def test_gateway_policy_can_block_ner_person_detection():
    class PersonRecognizer:
        def recognize(self, text):
            return [EntitySpan("PERSON", 0, len(text))]

    class NoCallProvider:
        called = False

        async def complete(self, prompt):
            self.called = True
            return prompt

        async def stream(self, prompt):
            yield prompt

    provider = NoCallProvider()
    gateway = Gateway(provider, Policy(rules={"PERSON": "block"}), DetectionPipeline([PersonRecognizer()]).detect)
    result = asyncio.run(gateway.process("Mehmet Demir"))
    assert result["blocked"] is True
    assert provider.called is False


def test_wiki_location_label_does_not_get_mislabeled_as_address():
    class LocationRecognizer:
        def recognize(self, text):
            return [EntitySpan("GPE", 0, len(text))]

    item = DetectionPipeline([LocationRecognizer()]).detect("Ankara")
    assert item[0].type == "LOCATION"
    assert Policy().action_for("LOCATION") == "mask"
    assert Policy().action_for("ADDRESS") == "mask"


def test_huggingface_ner_adapter_maps_aggregated_entity_offsets():
    recognizer = HuggingFaceNERRecognizer(
        lambda _text: [
            {"entity_group": "PERSON", "start": 0, "end": 11, "score": 0.93},
            {"entity_group": "LABEL_0", "start": 12, "end": 17, "score": 0.99},
            {"entity_group": "ORG", "start": 18, "end": 25, "score": 0.4},
        ],
        min_score=0.5,
    )
    found = DetectionPipeline([recognizer]).detect("Ayşe Yılmaz Ankara Örnek AŞ")
    assert [(item.type, item.value) for item in found] == [("PERSON", "Ayşe Yılmaz")]


def test_local_ner_loader_uses_offline_weights_and_word_level_predictor(monkeypatch):
    from turkce_llm_dlp import ner as ner_module

    calls = {}

    class Tokenizer:
        is_fast = True

    def load_tokenizer(path, **kwargs):
        calls["tokenizer"] = (path, kwargs)
        return Tokenizer()

    def load_model(path, **kwargs):
        calls["model"] = (path, kwargs)
        return object()

    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        AutoModelForTokenClassification=SimpleNamespace(from_pretrained=load_model),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(ner_module, "WordLevelPredictor", lambda tok, model, device: ("word-level", tok, model, device))
    predictor = _create_local_predictor("/tmp/local-model", -1)
    assert calls["tokenizer"][1]["local_files_only"] is True
    assert calls["model"][1]["local_files_only"] is True
    assert predictor[0] == "word-level"


def test_word_level_predictor_labels_whole_words_and_windows_long_text():
    torch = pytest.importorskip("torch")
    from turkce_llm_dlp.ner import WordLevelPredictor

    labels = {0: "O", 1: "B-ORG", 2: "I-ORG", 3: "B-PERSON"}

    class Encoded(dict):
        def __init__(self, ids, word_ids):
            super().__init__(input_ids=torch.tensor([ids]))
            self._word_ids = word_ids

        def word_ids(self, _index):
            return self._word_ids

    class Tokenizer:
        model_max_length = 512

        def __call__(self, words, **_kwargs):
            # Her sözcük 2 alt parça: ilk parça sözcüğün etiketini taşır; "'in" gibi ek parçası ikinci parçadır
            ids, word_ids = [0], [None]
            vocab = {"Kardeşler": 1, "Marketi'nin": 2, "Ayşe": 3}
            for index, word in enumerate(words):
                ids += [vocab.get(word, 5), 99]
                word_ids += [index, index]
            return Encoded(ids + [0], word_ids + [None])

    class Model(torch.nn.Module):
        config = SimpleNamespace(id2label=labels)

        def forward(self, input_ids):
            logits = torch.zeros((1, input_ids.shape[1], 4))
            for position, token in enumerate(input_ids[0].tolist()):
                # ilk alt parça sözcüğün etiketi (1 B-ORG, 2 I-ORG, 3 B-PERSON); ek parçası (99) bilinçli olarak
                # B-PERSON: eğitimde hiç etiketlenmediği için üretimde dikkate alınmamalı
                label = 3 if token == 99 else (token if token in (1, 2, 3) else 0)
                logits[0, position, label] = 5.0
            return SimpleNamespace(logits=logits)

    text = "akşam Kardeşler Marketi'nin önünde Ayşe ile buluşuruz"
    predictor = WordLevelPredictor(Tokenizer(), Model(), window_words=4, overlap_words=2)
    rows = predictor(text)
    found = [(r["entity_group"], text[r["start"] : r["end"]]) for r in rows]
    assert found == [("ORG", "Kardeşler Marketi'nin"), ("PERSON", "Ayşe")]


def test_ner_inference_timeout_terminates_worker_and_uses_rules(monkeypatch, tmp_path, caplog):
    class Connection:
        polls = 0
        closed = False

        def poll(self, _timeout):
            self.polls += 1
            return self.polls == 1  # startup ready; inference never completes

        def recv(self):
            return ("ready", None)

        def send(self, _text):
            pass

        def close(self):
            self.closed = True

    class ChildConnection:
        def close(self):
            pass

    class Process:
        terminated = False

        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

        def is_alive(self):
            return not self.terminated

        def terminate(self):
            self.terminated = True

        def join(self, timeout=None):
            pass

        def close(self):
            pass

    class Context:
        def __init__(self):
            self.parent = Connection()
            self.process = None

        def Pipe(self, duplex=True):
            return self.parent, ChildConnection()

        def Process(self, **kwargs):
            self.process = Process(**kwargs)
            return self.process

    context = Context()
    monkeypatch.setattr("turkce_llm_dlp.ner.multiprocessing.get_context", lambda _method: context)
    recognizer = ProcessIsolatedHuggingFaceNERRecognizer(
        str(tmp_path), startup_timeout_seconds=1, inference_timeout_seconds=0.01
    )
    detections = DetectionPipeline([recognizer]).detect("TCKN 10000000146")
    assert [item.type for item in detections] == ["TCKN"]
    assert context.process.terminated is True
    assert recognizer._disabled is True
    assert "10000000146" not in caplog.text
    recognizer.close()


def test_ner_worker_restarts_after_timeout_and_recovers(monkeypatch, tmp_path):
    import time as _time

    class Connection:
        def __init__(self, hang):
            self.hang = hang
            self.sent = None
            self.ready_sent = False

        def poll(self, _timeout):
            if not self.ready_sent:
                return True  # yükleme tamam
            return not self.hang  # ilk süreç yanıt vermez, ikincisi verir

        def recv(self):
            if not self.ready_sent:
                self.ready_sent = True
                return ("ready", None)
            return ("result", [{"entity_group": "PER", "start": 10, "end": 16, "score": 0.9}])

        def send(self, text):
            self.sent = text

        def close(self):
            pass

    class Child:
        def close(self):
            pass

    class Process:
        def __init__(self, **_kwargs):
            self.alive = True

        def start(self):
            pass

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.alive = False

        def kill(self):
            self.alive = False

        def join(self, timeout=None):
            pass

        def close(self):
            pass

    class Context:
        spawned = 0

        def Pipe(self, duplex=True):
            Context.spawned += 1
            return Connection(hang=Context.spawned == 1), Child()

        def Process(self, **kwargs):
            return Process(**kwargs)

    monkeypatch.setattr("turkce_llm_dlp.ner.multiprocessing.get_context", lambda _method: Context())
    recognizer = ProcessIsolatedHuggingFaceNERRecognizer(
        str(tmp_path), startup_timeout_seconds=1, inference_timeout_seconds=0.01, restart_delays=(0.05,)
    )
    pipeline = DetectionPipeline([recognizer])
    text = "selam ben atilla tc 10000000146"
    detections, degraded = pipeline.detect_with_status(text)
    assert degraded and [d.type for d in detections] == ["TCKN"] and not recognizer.available
    for _ in range(100):
        if recognizer.available:
            break
        _time.sleep(0.02)
    assert recognizer.available and recognizer.restarts == 1  # model kalıcı olarak kapanmadı
    detections, degraded = pipeline.detect_with_status(text)
    assert not degraded and ("PERSON", "atilla") in [(d.type, d.value) for d in detections]
    recognizer.close()


def test_detect_reports_ner_status(monkeypatch):
    from turkce_llm_dlp import api as api_module

    def broken(_text):
        raise RuntimeError("model çöktü")

    client = TestClient(app)
    monkeypatch.setattr(api_module, "detection_pipeline", DetectionPipeline([HuggingFaceNERRecognizer(broken)]))
    body = client.post("/v1/detect", json={"text": "TCKN 10000000146"}).json()
    assert body["ner"] == "unavailable" and body["detections"][0]["type"] == "TCKN"
    monkeypatch.setattr(api_module, "detection_pipeline", DetectionPipeline([HuggingFaceNERRecognizer(lambda _t: [])]))
    assert client.post("/v1/detect", json={"text": "merhaba"}).json()["ner"] == "ok"


def test_training_jsonl_rejects_empty_sentences(tmp_path):
    data = tmp_path / "empty.jsonl"
    data.write_text(json.dumps({"tokens": [], "ner_tags": []}) + "\n", encoding="utf-8")
    try:
        read_jsonl(data)
    except ValueError as exc:
        assert "boş cümle" in str(exc)
    else:
        raise AssertionError("empty NER sentence should be rejected")


def test_ner_runtime_error_falls_back_to_rules_without_logging_input(caplog):
    class BrokenRecognizer:
        def recognize(self, text):
            raise RuntimeError(f"model error on {text}")

    detections = DetectionPipeline([BrokenRecognizer()]).detect("TCKN 10000000146")
    assert [item.type for item in detections] == ["TCKN"]
    assert "10000000146" not in caplog.text
    assert "NER tanıyıcısı başarısız" in caplog.text


def test_invalid_ner_offsets_fail_closed_before_upstream():
    class InvalidRecognizer:
        def recognize(self, _text):
            return [EntitySpan("PERSON", -1, 3)]

    class Provider:
        called = False

        async def complete(self, prompt):
            self.called = True
            return prompt

        async def stream(self, prompt):
            yield prompt

    provider = Provider()
    gateway = Gateway(provider, detector=DetectionPipeline([InvalidRecognizer()]).detect)
    try:
        asyncio.run(gateway.process("Ayşe"))
    except ValueError as exc:
        assert "offset" in str(exc)
    else:
        raise AssertionError("invalid NER span should fail closed")
    assert provider.called is False


def test_api_ner_model_is_optional_and_load_failure_falls_back(monkeypatch, caplog):
    from turkce_llm_dlp.gazetteer import FacilityRecognizer, LocationGazetteer
    from turkce_llm_dlp.rules_ner import RuleEntityRecognizer

    def only_rules(detector):
        return [type(r) for r in detector.recognizers] == [RuleEntityRecognizer, LocationGazetteer, FacilityRecognizer]

    monkeypatch.delenv("TURKCE_DLP_NER_MODEL_DIR", raising=False)
    assert only_rules(api_module._configured_detector())
    monkeypatch.setenv("TURKCE_DLP_GAZETTEER", "0")
    assert [type(r) for r in api_module._configured_detector().recognizers] == [RuleEntityRecognizer]
    monkeypatch.setenv("TURKCE_DLP_RULE_NER", "0")
    assert api_module._configured_detector().recognizers == ()
    monkeypatch.delenv("TURKCE_DLP_RULE_NER")
    monkeypatch.delenv("TURKCE_DLP_GAZETTEER")

    monkeypatch.setenv("TURKCE_DLP_NER_MODEL_DIR", "/private/model-path")

    def fail_to_load(cls, _path, **_kwargs):
        raise OSError("private model path and request data")

    monkeypatch.setattr(HuggingFaceNERRecognizer, "from_local_model", classmethod(fail_to_load))
    assert only_rules(api_module._configured_detector())
    assert "/private/model-path" not in caplog.text
    assert "request data" not in caplog.text


def test_secret_rules_cover_vendor_tokens_passwords_entropy_and_pem():
    high_entropy = "Ab3dEf6hIj9Kl2MnOp5Qr8StUv1Wx4Yz"
    text = (
        f"api_key='{high_entropy}' AWS_SECRET_ACCESS_KEY={high_entropy} password='kisa' "
        "api_key=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa "
        "token=ghp_abcdefghijk123456 "
        "AIzaabcdefghijklmnopqrstuvwxyz123456789 "
        "-----BEGIN PRIVATE KEY-----\nnot-a-real-key-material\n-----END PRIVATE KEY-----"
    )
    matches = [item for item in detect(text) if item.type == "SECRET"]
    values = [item.value for item in matches]
    assert high_entropy in values
    assert values.count(high_entropy) == 2
    assert "kisa" in values
    assert "ghp_abcdefghijk123456" in values
    assert "AIzaabcdefghijklmnopqrstuvwxyz123456789" in values
    assert any(value.startswith("-----BEGIN PRIVATE KEY-----") for value in values)
    assert "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" not in values
    assert shannon_entropy("aaaaaaaa") == 0
    assert shannon_entropy(high_entropy) >= 3.5


def test_synthetic_address_generator_uses_schema_and_split_specific_data():
    train = synthetic_address_rows("train", 4, 11)
    test = synthetic_address_rows("test", 4, 11)
    allowed = {"O", "B-PERSON", "I-PERSON", "B-ORG", "I-ORG", "B-ADDRESS", "I-ADDRESS"}
    assert all(len(row["tokens"]) == len(row["ner_tags"]) for row in train + test)
    assert any(tag.startswith("B-ADDRESS") for row in train for tag in row["ner_tags"])
    assert all(set(row["ner_tags"]) <= allowed for row in train + test)
    assert {tuple(row["tokens"]) for row in train}.isdisjoint({tuple(row["tokens"]) for row in test})


def test_prepare_data_maps_wiki_labels_and_writes_splits(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    for split, text in {
        "train": "Ayşe B-PER\nAnkara B-GPE\nAcme B-ORG\nroman B-MISC\n\n",
        "dev": "Mehmet B-PER\n\n",
        "test": "İzmir B-GPE\n\n",
    }.items():
        (wiki / f"{split}.conll").write_text(text, encoding="utf-8")
    output = tmp_path / "out"
    counts = prepare(wiki, output, address_count=2, seed=7)
    assert counts == {"train": 3, "dev": 3, "test": 3}
    quality_report = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
    assert quality_report["splits"]["train"]["rows"] == 3
    assert quality_report["splits"]["train"]["entity_counts"]["LOCATION"] == 1
    assert quality_report["splits"]["train"]["source_labels"]["ignored_entity_tag_counts"] == {"MISC": 1}
    assert "Ayşe" not in (output / "quality_report.json").read_text(encoding="utf-8")
    train = [json.loads(line) for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert train[0]["ner_tags"] == ["B-PERSON", "B-LOCATION", "B-ORG", "O"]
    assert train[1]["source"] == "synthetic-address-v1"


def test_ner_adjacent_address_parts_are_merged():
    from turkce_llm_dlp.ner import DetectionPipeline, EntitySpan

    text = "Adres: Lale Sokak No: 5, Örnek Mahallesi. Kişi: Ali, Veli"
    first = text.index("Lale")
    comma = text.index(",")
    second_end = text.index(".")
    ali = text.index("Ali")
    veli = text.index("Veli")

    class Split:
        def recognize(self, _text):
            return [
                EntitySpan("ADDRESS", first, comma),
                EntitySpan("ADDRESS", comma + 2, second_end),
                EntitySpan("PERSON", ali, ali + 3),
                EntitySpan("PER", veli, veli + 4),
            ]

    found = [(d.type, d.value) for d in DetectionPipeline([Split()]).detect(text)]
    assert ("ADDRESS", "Lale Sokak No: 5, Örnek Mahallesi") in found
    # Kişiler birleştirilmez: iki ayrı kişi iki ayrı placeholder almalı.
    assert ("PERSON", "Ali") in found and ("PERSON", "Veli") in found


def test_ner_semicolon_separated_addresses_stay_separate():
    from turkce_llm_dlp.ner import EntitySpan, merge_adjacent_spans

    text = "Lale Sk. No:2 Bağcılar; Erguvan Sk. No:17 Sancaktepe"
    cut = text.index(";")
    spans = [EntitySpan("ADDRESS", 0, cut), EntitySpan("ADDRESS", cut + 2, len(text))]
    assert merge_adjacent_spans(text, spans) == spans


def test_address_trailing_sentence_period_is_trimmed_but_abbreviation_kept():
    from turkce_llm_dlp.ner import EntitySpan, trim_address_span

    text = "Teslimat adresi: Lale Sokak No: 5, Örnek Mahallesi."
    start = text.index("Lale")
    trimmed = trim_address_span(text, EntitySpan("ADDRESS", start, len(text)))
    assert text[trimmed.start : trimmed.end] == "Lale Sokak No: 5, Örnek Mahallesi"

    text = "Adres: Örnek Mah. Lale Sk."
    start = text.index("Örnek")
    kept = trim_address_span(text, EntitySpan("ADDRESS", start, len(text)))
    assert text[kept.start : kept.end] == "Örnek Mah. Lale Sk."
    person = EntitySpan("PERSON", 0, 4)
    assert trim_address_span(text, person) == person


def test_semicolon_inside_model_span_does_not_merge_two_addresses():
    from turkce_llm_dlp.ner import DetectionPipeline, EntitySpan

    text = "Teslimat listesi: Lale Sk. No:2 Bağcılar; Erguvan Sk. No:17 Sancaktepe"
    first = text.index("Lale")
    cut = text.index(";")
    second = text.index("Erguvan")

    class GreedyModel:
        def recognize(self, _text):
            # Model ilk adresin sonuna ";" işaretini katıyor.
            return [EntitySpan("ADDRESS", first, cut + 1), EntitySpan("ADDRESS", second, len(text))]

    values = [d.value for d in DetectionPipeline([GreedyModel()]).detect(text) if d.type == "ADDRESS"]
    assert values == ["Lale Sk. No:2 Bağcılar", "Erguvan Sk. No:17 Sancaktepe"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("tel 0532 123 45 67", "0532 123 45 67"),
        ("beni ara: +90 532 123 45 67", "+90 532 123 45 67"),
        ("telefon 05321234567", "05321234567"),
        ("sabit (0212) 555 12 34", "(0212) 555 12 34"),
        ("(212) 555 12 34 merkez", "(212) 555 12 34"),
        ("iletişim: 0312-444-12-34", "0312-444-12-34"),
        ("cep: 532 123 45 67", "532 123 45 67"),
        ("whatsapp 5321234567", "5321234567"),
        ("0090 532 1234567", "0090 532 1234567"),
        ("tel 05XX 000 00 01.", "05XX 000 00 01"),  # kısmen gizlenmiş numara da maskelenir
        ("geldiğinde ararmısın 5477658697", "5477658697"),
        ("numaram 5321234567 yaz", "5321234567"),
        ("ofis hattı 2125551234 arayın", "2125551234"),
        ("hey 2125462345", "2125462345"),
        ("bizim ev 3122345678", "3122345678"),
    ],
)
def test_phone_numbers_are_detected(text, expected):
    assert [(d.type, d.value) for d in detect(text)] == [("PHONE", expected)]


@pytest.mark.parametrize(
    "text",
    [
        "sipariş no 5321234567",
        "hesap no 5321234567",
        "fiş 5021234567 ödendi",
        "toplam 5000000000 TL",
        "kod 1234567890",  # alan kodu değil
        "sipariş no 532 123 45 67",
        "takip 4123456789",
        "fatura no 0123456789",
        "tarih 2026-10-12",
        "tutar 312.500.000 TL",
        "IP 192.168.10.12",
        "oda 532 kat 3",
        "XXX XXX XX XX",
    ],
)
def test_phone_rule_avoids_numeric_false_positives(text):
    assert [d for d in detect(text) if d.type == "PHONE"] == []


def test_same_phone_in_different_formats_gets_one_placeholder():
    from turkce_llm_dlp.masking import MaskingSession

    masked, _ = MaskingSession().mask("tel 0532 123 45 67, yine +90 (532) 123-45-67")
    assert masked.count("<PHONE_1>") == 2


def test_model_span_drops_apostrophe_suffix_so_same_person_gets_one_placeholder():
    from turkce_llm_dlp.masking import MaskingSession
    from turkce_llm_dlp.ner import DetectionPipeline, EntitySpan

    text = "Ayşe Demir'e söyle, Ayşe Demir'in dosyası hazır"

    class Model:
        def recognize(self, _text):
            first, second = text.index("Ayşe"), text.rindex("Ayşe")
            return [
                EntitySpan("PERSON", first, first + len("Ayşe Demir'e")),
                EntitySpan("PERSON", second, second + len("Ayşe Demir'in")),
            ]

    found = DetectionPipeline([Model()]).detect(text)
    assert [d.value for d in found] == ["Ayşe Demir", "Ayşe Demir"]
    masked, _ = MaskingSession().mask(text, found)
    assert masked == "<PERSON_1>'e söyle, <PERSON_1>'in dosyası hazır"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("selam ben atilla tc 11028365789", True),  # kontrol haneleri tutmuyor ama TC ipucu var
        ("T.C. Kimlik No: 11028365789", True),
        ("tckn=11028365789", True),
        ("sipariş no 11028365789", False),
        ("11028365789 numaralı fatura", False),
        ("etc 11028365789", False),
    ],
)
def test_tckn_with_context_is_masked_even_if_checksum_fails(text, expected):
    assert any(d.type == "TCKN" for d in detect(text)) is expected


def test_ner_keepalive_warms_idle_worker_and_restarts_if_it_hangs(monkeypatch, tmp_path):
    import time as _time

    sent = []

    class Connection:
        def __init__(self, hang):
            self.hang = hang
            self.ready_sent = False

        def poll(self, _timeout):
            return not self.ready_sent or not self.hang

        def recv(self):
            if not self.ready_sent:
                self.ready_sent = True
                return ("ready", None)
            return ("result", [])

        def send(self, text):
            sent.append((self.hang, text))

        def close(self):
            pass

    class Child:
        def close(self):
            pass

    class Process:
        alive = True

        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.alive = False

        kill = terminate

        def join(self, timeout=None):
            pass

        def close(self):
            pass

    class Context:
        spawned = 0

        def Pipe(self, duplex=True):
            Context.spawned += 1
            return Connection(hang=Context.spawned == 1), Child()

        def Process(self, **kwargs):
            return Process(**kwargs)

    monkeypatch.setattr("turkce_llm_dlp.ner.multiprocessing.get_context", lambda _method: Context())
    recognizer = ProcessIsolatedHuggingFaceNERRecognizer(
        str(tmp_path),
        startup_timeout_seconds=1,
        inference_timeout_seconds=0.01,
        restart_delays=(0.05,),
        keepalive_seconds=0.05,
    )
    for _ in range(150):
        if recognizer.restarts and any(not hang for hang, _ in sent):
            break
        _time.sleep(0.02)
    texts = {text for _, text in sent}
    assert texts == {ProcessIsolatedHuggingFaceNERRecognizer.KEEPALIVE_TEXT}  # yalnız deneme metni, kullanıcı metni yok
    assert recognizer.restarts == 1 and recognizer.available  # takılan süreç kullanıcı mesajından önce yenilendi
    recognizer.close()
