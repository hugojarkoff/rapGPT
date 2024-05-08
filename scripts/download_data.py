# Make sure your kaggle.json API token is correctly located in ~/.kaggle dir

from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

DATA_PATH = Path("data")
DATA_PATH.mkdir(exist_ok=True, parents=True)


def main():
    api = KaggleApi()
    api.dataset_download_files(
        dataset="adibhabbou/french-rap-lyrics",
        path=DATA_PATH,
        unzip=True,
        force=True,
    )


if __name__ == "__main__":
    main()
