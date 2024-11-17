import os
import argparse
import random
import torch
from rapgpt.config import Config
from rapgpt.encoder import Encoder
from rapgpt.model import HFHubTransformerModel, TransformerModel
import gradio as gr
from huggingface_hub import hf_hub_download

HF_REPO = "hugojarkoff/rapgpt"


def valid_file(filepath: str) -> str:
    """Custom argparse type to check if a file exists."""
    if not os.path.isfile(filepath):
        raise argparse.ArgumentTypeError(
            f"'{filepath}' is not a valid file path or does not exist."
        )
    return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local inference using Gradio app")

    parser.add_argument(
        "--config-file",
        type=valid_file,
        default=None,
        help="Path to config file used for training (optional). If not specified, will use the file from HF hugojarkoff/rapgpt",
    )
    parser.add_argument(
        "--artists-tokens",
        type=valid_file,
        default=None,
        help="Path to artists/tokens hashmap file generated in training (optional). If not specified, will use the file from HF hugojarkoff/rapgpt",
    )
    parser.add_argument(
        "--local-checkpoint",
        type=valid_file,
        default=None,
        help="Path to the local checkpoint for inference (optional). If not specified, will use the file from hugojarkoff/rapgpt",
    )

    args = parser.parse_args()

    if args.artists_tokens:
        artists_tokens = args.artists_tokens
    else:
        artists_tokens = hf_hub_download(
            repo_id=HF_REPO,
            filename="artists_tokens.txt",
            repo_type="model",
        )

    with open(artists_tokens, "r") as f:
        artists_tokens = {
            line.split(":")[0]: int(line.split(":")[1].rstrip("\n")) for line in f
        }

    if args.config_file:
        artists_tokens = args.config_file
    else:
        config_file = hf_hub_download(
            repo_id=HF_REPO, filename="config.toml", repo_type="model"
        )

    config = Config.load_from_toml(config_file)
    encoder = Encoder(config=config)

    if args.local_checkpoint:
        model = TransformerModel.load_state_dict(args.local_checkpoint)
    else:
        model = HFHubTransformerModel.from_pretrained(HF_REPO)

    def predict(
        lyrics_prompt: str,
        new_tokens: int,
        artist_token: int,
        seed: int = 42,
    ):
        # Set Seed
        random.seed(seed)
        torch.manual_seed(seed)

        # Predict
        sample_input = encoder.encode_data(lyrics_prompt)
        sample_input = torch.tensor(sample_input).unsqueeze(0)
        output = model.generate(
            x=sample_input,
            new_tokens=new_tokens,
            artist_token=artist_token,
        )
        return encoder.decode_data(output[0].tolist())

    gradio_app = gr.Interface(
        predict,
        inputs=[
            gr.Textbox(
                value="ekip",
                label="Lyrics prompt",
                info="rapGPT will continue this prompt",
            ),
            gr.Number(
                value=100,
                maximum=100,
                label="New tokens to generate",
                info="Number of new tokens to generate (limited to 100)",
            ),
            gr.Dropdown(
                value="freeze corleone",
                choices=artists_tokens.keys(),
                type="index",
                label="Artist",
                info="Which artist style to generate",
            ),
            gr.Number(
                value=42, label="Random seed", info="Change for different results"
            ),
        ],
        outputs=[gr.TextArea(label="Generated Lyrics")],
        title="rapGPT",
    )

    gradio_app.launch()
