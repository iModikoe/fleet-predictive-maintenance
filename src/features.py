import pandas as pd

SENSORS = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21"]
BASE_FEATURES = ["cycle", "setting_1", "setting_2", "setting_3", *SENSORS]


def engineer_features(frame: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    data = frame.sort_values(["unit_number", "cycle"]).copy()
    grouped = data.groupby("unit_number", group_keys=False)
    for sensor in SENSORS:
        data[f"{sensor}_mean_{window}"] = grouped[sensor].transform(lambda values: values.rolling(window, min_periods=1).mean())
        data[f"{sensor}_std_{window}"] = grouped[sensor].transform(lambda values: values.rolling(window, min_periods=1).std()).fillna(0)
        data[f"{sensor}_trend_{window}"] = data[sensor] - grouped[sensor].shift(window - 1).fillna(data[sensor])
    return data


def model_features(frame: pd.DataFrame) -> list[str]:
    excluded = {"unit_number", "rul"}
    return [column for column in frame.columns if column not in excluded]
