from rapgpt.trainer import Trainer
from rapgpt.config import Config
import argparse


def main():
    parser = argparse.ArgumentParser(description="Training args.")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to toml config file",
        default="configs/config.toml",
    )
    args = parser.parse_args()
    config = Config.load_from_toml(args.config)
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
