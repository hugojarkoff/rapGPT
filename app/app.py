import random
import torch
from rapgpt.config import Config
from rapgpt.encoder import Encoder
from rapgpt.model import TransformerModel
import gradio as gr

CONFIG = "configs/config.toml"
ARTISTS_TOKENS_FILE = "checkpoints/fiery-spaceship-65/artists_tokens.txt"
CHECKPOINT = "checkpoints/fiery-spaceship-65/model.pt"


with open(ARTISTS_TOKENS_FILE, "r") as f:
    artists_tokens = {
        line.split(":")[0]: int(line.split(":")[1].rstrip("\n")) for line in f
    }

config = Config.load_from_toml(CONFIG)
encoder = Encoder(config=config)

model = TransformerModel(
    vocab_size=encoder.vocab_size, artists_size=len(artists_tokens), config=config
)
model.load_state_dict(
    torch.load(CHECKPOINT, weights_only=True, map_location=torch.device("cpu"))
)


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
        gr.Textbox(placeholder="ekip"),
        gr.Number(value=100),
        gr.Dropdown(
            value="freeze corleone", choices=artists_tokens.keys(), type="index"
        ),
        gr.Number(value=42),
    ],
    outputs=[gr.TextArea()],
    title="rapGPT",
)

if __name__ == "__main__":
    gradio_app.launch()
