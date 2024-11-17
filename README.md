# rapGPT

Train a GPT-like model to generate French rap lyrics.

Mostly a fun and educational personal project, learning how to design and train a GPT-like architecture from scratch.

## 0. Dependencies management
This project uses [Rye](https://rye-up.com/). Make sure it's installed in your system.

To install **all** dependencies (downloading data, training etc.), run `rye sync --all-features` in project directory.

## 1. Data
This project uses [French Rap Lyrics Kaggle dataset](https://www.kaggle.com/datasets/adibhabbou/french-rap-lyrics?resource=download).

To download it, register your kaggle API token. See instructions [here](https://www.kaggle.com/docs/api). Basically simply download and move your `kaggle.json` token to `~/.kaggle/kaggle.json`.

Then run `python scripts/download_data.py`.

## 2. Train
Make sure you have access to a decent GPU, as the default model config is pretty VRAM-heavy.

From repo root, run `python scripts/train.py` with an optional `config` arg (by default pointed to `configs/config.toml`).

The best model is tracked and saved on disk by [`torcheval.metrics.Perplexity`](https://pytorch.org/torcheval/main/generated/torcheval.metrics.Perplexity.html). By default, checkpoints are saved in `checkpoints/<run_name>`.

**NOTE**: This project uses [WandB](https://wandb.ai/) to log and record experiments. If your training config specifies `wandb.mode = online`, make sure you've registered your account with your API key.

## 3. Pushing to HF Hub
Once your model is trained, you can push the checkpoint to [HF](https://huggingface.co/) using `scripts/push_to_hf_hub.py` with the correct specified arguments. It will push the following three components:
- `model.pt` (specified argument), converted to `model.safetensors` (using the `rapgpt.model.HFHubTransformerModel` mixin) for ease of inference on HF Space;
- `config.toml` (specified argument);
- `artists_tokens.txt` (specified argument).

These three components are required for inference (see next section).

## 4. Local Inference
This project uses [Gradio](https://www.gradio.app/) for local and online inference.

Local inference is done using `python app/app.py` script. Some additional arguments can be passed, essentially indicating wether to use the [default checkpoint on HF Hub](https://huggingface.co/hugojarkoff/rapGPT/tree/main) or some local checkpoint.

## 5. Online Inference
Online inference is served [on HF](https://huggingface.co/spaces/hugojarkoff/rapGPT) through the (more or less) same Gradio `app`. It automatically calls the [default checkpoint on HF Hub](https://huggingface.co/hugojarkoff/rapGPT/tree/main) for inference.

## Future Works / Ideas

Contribs welcome!

- Retrain a bigger model for even better / more style-accurate lyrics
- Find a way to select multiple artists tokens (for mixing styles)

## Credits

Inspired by the great [nanoGPT](https://github.com/karpathy/nanoGPT)

