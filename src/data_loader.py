from pathlib import Path
import pandas as pd

COLUMNS = ["unit_number", "cycle", "setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 22)]


def load_trajectory(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    if frame.empty or frame[COLUMNS].isna().any().any():
        raise ValueError("Trajectory file is empty or malformed")
    return frame


def add_training_rul(frame: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    result = frame.copy()
    maximum_cycle = result.groupby("unit_number")["cycle"].transform("max")
    result["rul"] = (maximum_cycle - result["cycle"]).clip(upper=cap)
    return result


def load_test_targets(path: str | Path) -> pd.Series:
    return pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0].rename("rul")
