from rapgpt.trainer import Trainer
from rapgpt.config import Config

if __name__ == "__main__":
    config = Config()
    trainer = Trainer(config)
    trainer.train()
