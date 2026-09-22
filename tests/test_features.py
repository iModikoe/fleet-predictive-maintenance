import pandas as pd
from src.data_loader import add_training_rul
from src.features import engineer_features, model_features


def test_rul_is_calculated_within_each_engine():
    frame = pd.DataFrame({"unit_number": [1, 1, 1, 2, 2], "cycle": [1, 2, 3, 1, 2]})
    assert add_training_rul(frame)["rul"].tolist() == [2, 1, 0, 1, 0]


def test_engine_id_is_not_a_model_feature():
    frame = pd.DataFrame({"unit_number": [1], "cycle": [1], "rul": [4], "sensor_2": [1.0]})
    assert "unit_number" not in model_features(frame)
    assert "rul" not in model_features(frame)
