from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
RAW_DIR = Path(__file__).resolve().parents[1] / "data/raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with urlopen(URL) as response:
        outer = ZipFile(BytesIO(response.read()))
    nested_name = next(name for name in outer.namelist() if name.endswith("CMAPSSData.zip"))
    inner = ZipFile(BytesIO(outer.read(nested_name)))
    for name in ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt", "readme.txt"]:
        (RAW_DIR / name).write_bytes(inner.read(name))
    print(f"Saved FD001 data to {RAW_DIR}")


if __name__ == "__main__":
    main()
