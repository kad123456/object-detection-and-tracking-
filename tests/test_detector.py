import pytest

from src.detector import Detector


def test_class_names_and_ids_resolve_case_insensitively():
    detector = Detector.__new__(Detector)
    detector.names = {0: "person", 2: "car"}
    detector.model_path = "test-model.pt"

    assert detector.resolve_classes(["PERSON", "2", "car"]) == [0, 2]
    assert detector.resolve_classes([]) is None


def test_unknown_class_is_reported():
    detector = Detector.__new__(Detector)
    detector.names = {0: "person"}
    detector.model_path = "test-model.pt"

    with pytest.raises(ValueError, match="Unknown class"):
        detector.resolve_classes(["spaceship"])
