import numpy as np

from traffic_bert.calibration import calibrate_thresholds


def test_calibrate_thresholds_selects_per_label_values() -> None:
    y_true = np.array(
        [
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
        ]
    )
    y_prob = np.array(
        [
            [0.9, 0.1],
            [0.7, 0.4],
            [0.2, 0.8],
            [0.1, 0.6],
        ]
    )

    thresholds, results = calibrate_thresholds(
        y_true,
        y_prob,
        labels=["a", "b"],
        candidate_thresholds=np.array([0.3, 0.5, 0.7]),
    )

    assert set(thresholds) == {"a", "b"}
    assert thresholds["a"] == 0.7
    assert thresholds["b"] == 0.5
    assert all(result.f1 == 1.0 for result in results)


def test_calibrate_thresholds_keeps_default_without_support() -> None:
    y_true = np.array([[0], [0]])
    y_prob = np.array([[0.2], [0.8]])

    thresholds, results = calibrate_thresholds(y_true, y_prob, labels=["empty"])

    assert thresholds["empty"] == 0.5
    assert results[0].support == 0

