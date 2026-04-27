import torch

from traffic_bert.models.neural_baselines import (
    ByteCnnClassifier,
    ByteGruClassifier,
    ByteTransformerClassifier,
)


def test_neural_baseline_shapes() -> None:
    input_ids = torch.randint(0, 32, (2, 2, 8))
    attention_mask = torch.ones_like(input_ids)

    models = [
        ByteCnnClassifier(vocab_size=32, num_labels=3, hidden_size=16),
        ByteGruClassifier(vocab_size=32, num_labels=3, hidden_size=16),
        ByteTransformerClassifier(
            vocab_size=32,
            num_labels=3,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            max_length=16,
        ),
    ]

    for model in models:
        logits = model(input_ids, attention_mask)
        assert logits.shape == (2, 3)

