import torch
from rapgpt.config import Config
from rapgpt.model import HFHubTransformerModel
from rapgpt.encoder import Encoder
import argparse
from huggingface_hub import HfApi

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

    api = HfApi()

    api.upload_file(
        path_or_fileobj=args.artists_tokens,
        path_in_repo="artists_tokens.txt",
        repo_id="hugojarkoff/rapgpt",
        repo_type="model",
    )

    with open(args.artists_tokens, "r") as f:
        artists_tokens = {
            line.split(":")[0]: int(line.split(":")[1].rstrip("\n")) for line in f
        }

    api.upload_file(
        path_or_fileobj=args.config_file,
        path_in_repo="config.toml",
        repo_id="hugojarkoff/rapgpt",
        repo_type="model",
    )

    config = Config.load_from_toml(args.config_file)

    encoder = Encoder(config=config)

    model = HFHubTransformerModel(
        vocab_size=encoder.vocab_size
        + 1,  # NOTE: Weird bug when loading checkpoint (vocab_size=50257), TODO: Fix
        artists_size=len(artists_tokens),
        **config.model.model_dump(),
    )

    model.load_state_dict(
        torch.load(args.checkpoint, weights_only=True, map_location=torch.device("cpu"))
    )

    model.push_to_hub("hugojarkoff/rapgpt")
