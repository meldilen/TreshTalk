import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

DATASETS = {
    "trashnet": "feyzazkefe/trashnet",
    "WaRP": "parohod/warp-waste-recycling-plant-dataset",
    # "CIFAR-100": "fedesoriano/cifar100",
    "12classes": "mostafaabla/garbage-classification"
}
RAW_DIR = Path("data/raw")

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    for name, slug in DATASETS.items():
        out_dir = RAW_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name} -> {out_dir}")
        api.dataset_download_files(slug, path=str(out_dir), unzip=True, quiet=False)
        print(f"✅ Done: {name}")

if __name__ == "__main__":
    main()
