from pathlib import Path

from traffic_bert.labels import LabelMap


def test_label_map_known_cic_labels() -> None:
    label_map = LabelMap.from_yaml(Path("configs/label_map.yaml"))

    target = label_map.map_source_label("DoS Hulk")

    assert target.major_label == "dos_ddos"
    assert target.minor_labels == ("dos_hulk",)
    assert label_map.major_id("dos_ddos") >= 0
    assert label_map.minor_multi_hot(["dos_hulk"])[label_map.minor_to_id["dos_hulk"]] == 1.0


def test_label_map_ustc_family_labels_before_generic_bot() -> None:
    label_map = LabelMap.from_yaml(Path("configs/label_map.yaml"))

    assert label_map.map_source_label("Htbot").minor_labels == ("htbot",)
    assert label_map.map_source_label("Bot").minor_labels == ("botnet",)
    assert label_map.map_source_label("Botnet").minor_labels == ("botnet",)
    assert label_map.map_source_label("Nsis-ay").minor_labels == ("nsis",)


def test_label_map_fallback() -> None:
    label_map = LabelMap.from_yaml(Path("configs/label_map.yaml"))

    target = label_map.map_source_label("unknown attack name")

    assert target.major_label == "other_attack"
    assert target.minor_labels == ("other_attack",)
