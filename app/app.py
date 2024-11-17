import argparse
import random
import torch
from rapgpt.config import Config
from rapgpt.encoder import Encoder
from rapgpt.model import HFHubTransformerModel
import gradio as gr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert and push locally trained and saved model to HF Hub"
    )

    parser.add_argument(
        "--config-file",
        default="configs/config.toml",
        help="Path to config file used for training.",
    )
    parser.add_argument(
        "--artists-tokens",
        default="checkpoints/fiery-spaceship-65/artists_tokens.txt",
        help="Path to artists/tokens hashmap file generated in training.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/fiery-spaceship-65/model.pt",
        help="Path to disk checkpoint of model.",
    )

    args = parser.parse_args()

    with open(args.artists_tokens, "r") as f:
        artists_tokens = {
            line.split(":")[0]: int(line.split(":")[1].rstrip("\n")) for line in f
        }

    config = Config.load_from_toml(args.config_file)
    encoder = Encoder(config=config)
    model = HFHubTransformerModel.from_pretrained("hugojarkoff/rapgpt")

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
            gr.Textbox(value="ekip"),
            gr.Number(value=100),
            gr.Dropdown(
                value="freeze corleone", choices=artists_tokens.keys(), type="index"
            ),
            gr.Number(value=42),
        ],
        outputs=[gr.TextArea()],
        title="rapGPT",
    )

    gradio_app.launch()
