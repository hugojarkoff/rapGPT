# rapGPT

Train a GPT-like model to generate French rap lyrics.

## 0. Dependencies management
This project uses [Rye](https://rye-up.com/). Make sure it's installed in your system.

To install **all** dependencies (downloading data, training etc.), run `rye sync --all-features` in project directory.

## 1. Data
This project uses [French Rap Lyrics Kaggle dataset](https://www.kaggle.com/datasets/adibhabbou/french-rap-lyrics?resource=download).

To download it, register your kaggle API token. See instructions [here](https://www.kaggle.com/docs/api). Basically simply download and move your `kaggle.json` token to `~/.kaggle/kaggle.json`.

Then run `python scripts/download_data.py`.
