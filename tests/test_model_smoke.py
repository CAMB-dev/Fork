import torch

from traffic_bert.labels import LabelMap
from traffic_bert.models import ByteBertForHierarchicalClassification, create_bert_config
from traffic_bert.tokenizer import ByteTokenizer


def test_byte_bert_classifier_forward_and_loss() -> None:
    label_map = LabelMap.from_yaml("configs/label_map.yaml")
    tokenizer = ByteTokenizer()
    config = create_bert_config(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=16,
    )
    model = ByteBertForHierarchicalClassification(
        bert_config=config,
        num_major_labels=len(label_map.major_labels),
        num_minor_labels=len(label_map.minor_labels),
        minor_to_major=label_map.minor_major_ids(),
        context_feature_size=7,
    )

    input_ids = torch.randint(0, tokenizer.vocab_size, (2, 2, 16))
    attention_mask = torch.ones_like(input_ids)
    window_mask = torch.ones(2, 2, dtype=torch.bool)
    context_features = torch.rand(2, 7)
    major_labels = torch.tensor([label_map.major_id("benign"), label_map.major_id("dos_ddos")])
    minor_labels = torch.zeros(2, len(label_map.minor_labels))
    minor_labels[1, label_map.minor_to_id["ddos"]] = 1.0

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        window_mask=window_mask,
        context_features=context_features,
        major_labels=major_labels,
        minor_labels=minor_labels,
    )

    assert outputs["major_logits"].shape == (2, len(label_map.major_labels))
    assert outputs["minor_logits"].shape == (2, len(label_map.minor_labels))
    assert outputs["classifier_repr"].shape[-1] == config.hidden_size * 2
    assert outputs["loss"].requires_grad
