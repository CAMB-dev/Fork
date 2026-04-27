from pathlib import Path

import torch

from traffic_bert.cli import _load_bert_encoder_from_mlm, _make_classifier
from traffic_bert.config import append_jsonl, load_yaml, set_seed, write_json
from traffic_bert.labels import LabelMap
from traffic_bert.models import create_mlm_model


def test_config_helpers_round_trip(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("a: 1\n", encoding="utf-8")
    json_path = tmp_path / "out.json"
    jsonl_path = tmp_path / "out.jsonl"

    assert load_yaml(config_path) == {"a": 1}
    write_json(json_path, {"ok": True})
    append_jsonl(jsonl_path, {"step": 1})

    assert '"ok": true' in json_path.read_text(encoding="utf-8")
    assert jsonl_path.read_text(encoding="utf-8").strip() == '{"step": 1}'


def test_seed_sets_torch_random_state() -> None:
    set_seed(123)
    first = torch.rand(1)
    set_seed(123)
    second = torch.rand(1)

    assert torch.equal(first, second)


def test_load_bert_encoder_from_mlm_checkpoint(tmp_path: Path) -> None:
    label_map = LabelMap.from_yaml("configs/label_map.yaml")
    mlm = create_mlm_model(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=16,
    )
    checkpoint = tmp_path / "mlm.pt"
    torch.save({"model_state_dict": mlm.state_dict()}, checkpoint)
    classifier = _make_classifier(
        label_map,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate_size=64,
        max_position_embeddings=16,
    )

    _load_bert_encoder_from_mlm(classifier, checkpoint)

    assert torch.equal(
        classifier.bert.embeddings.word_embeddings.weight,
        mlm.bert.embeddings.word_embeddings.weight,
    )

