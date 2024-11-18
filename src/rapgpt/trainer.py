import torch
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics import Perplexity
import random
import numpy as np
from rapgpt.model import TransformerModel
from rapgpt.config import Config
from rapgpt.encoder import Encoder
from rapgpt.data import Corpus, Lyrics
from rapgpt.callbacks import Checkpoint
from loguru import logger
from pathlib import Path
import wandb

loggable = str | int | float


class Trainer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.set_seed(self.config.training.seed)

        ## Logging Server
        run = wandb.init(
            project=config.wandb.project,
            mode=config.wandb.mode,
            tags=config.wandb.tags,
            group=config.wandb.group,
            config=config.model_dump(),
        )
        checkpoint_dir = Path(self.config.checkpoint.save_path) / run.name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        ## Dataset
        self.encoder = Encoder(config=self.config)
        self.corpus = Corpus(config=self.config, encoder=self.encoder)
        # Save artist tokens
        self.corpus.dump_artists_tokens(checkpoint_dir / "artists_tokens.txt")

        ## Model
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        vocab_size = self.encoder.vocab_size
        self.model = TransformerModel(
            vocab_size=vocab_size,
            artists_size=len(self.corpus.artists_tokens),
            **self.config.model.model_dump(),
        ).to(device=self.device)

        ## Loss Function and Optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.training.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, self.config.scheduler.gamma
        )
        self.perplexity = Perplexity(device=self.device)

        ## Checkpoint Callback
        self.checkpoint = Checkpoint(self.config, run.name)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def log(self, message: str | dict[str, loggable]) -> None:
        if isinstance(message, Lyrics):
            logger.info(message)
            columns = ["generated_lyrics"]
            data = [[message]]
            table = wandb.Table(columns=columns, data=data)
            wandb.log({"generated_lyrics": table})
            return
        if isinstance(message, str):
            logger.info(message)
            return
        assert isinstance(message, dict)
        for k, v in message.items():
            logger.info(f"{k}:{v}")
        wandb.log(message)

    def generate(self) -> str:
        # Encode the seed text
        sample_input = self.encoder.encode_data(self.config.evaluation.sample_text)
        sample_input = torch.tensor(sample_input).unsqueeze(0).to(self.device)

        output = self.model.generate(
            sample_input,
            self.config.evaluation.new_tokens,
            self.config.evaluation.artist_token,
        )

        # Decode the generated tokens
        return self.encoder.decode_data(output[0].tolist())

    @torch.no_grad()
    def evaluate(self) -> None:
        self.model.eval()

        val_loss = 0.0

        for _ in range(self.config.training.num_validation_steps):
            inputs, targets, artists = self.corpus.get_random_batch(
                batch_size=self.config.training.batch_size,
                block_size=self.config.model.context_length,
                split="val",
            )

            inputs, targets, artists = (
                inputs.long().to(self.device),
                targets.long().to(self.device),
                artists.long().to(self.device),
            )

            output = self.model(inputs, artists)

            # Loss computation
            output_flat = output.reshape(-1, output.shape[2])
            targets_flat = targets.reshape(-1)
            val_loss += self.loss_fn(output_flat, targets_flat)
            self.perplexity.update(output, targets)

        self.log({"val_loss": val_loss / self.config.training.num_validation_steps})

        perplexity = self.perplexity.compute().item()
        self.log({"val_perplexity": perplexity})
        self.perplexity.reset()

        generated_lyrics = self.generate()
        self.log(Lyrics(generated_lyrics))

        # Callback
        self.checkpoint.update(perplexity, self.model)

    def train(
        self,
    ) -> None:
        self.log("Training model")
        self.log(f"Trainable parameters: {self.model.learnable_params}")
        self.log({"lr": self.scheduler.get_last_lr()[0]})

        self.evaluate()

        # Training loop
        for step in range(self.config.training.num_training_steps):
            self.model.train()

            inputs, targets, artists = self.corpus.get_random_batch(
                batch_size=self.config.training.batch_size,
                block_size=self.config.model.context_length,
            )

            inputs, targets, artists = (
                inputs.long().to(self.device),
                targets.long().to(self.device),
                artists.long().to(self.device),
            )

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(inputs, artists)

            # Loss computation
            output_flat = output.reshape(-1, output.shape[2])  # Flatten output
            targets_flat = targets.reshape(-1)  # Flatten targets
            loss = self.loss_fn(output_flat, targets_flat)  # Compute loss
            self.perplexity.update(output, targets)

            self.log({"train_loss": loss})

            perplexity = self.perplexity.compute().item()
            self.log({"train_perplexity": perplexity})
            self.perplexity.reset()

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Evaluation loop every evaluation_cycle steps
            if step % self.config.training.evaluation_cycle == 0:
                self.evaluate()
                self.scheduler.step()
                self.log({"lr": self.scheduler.get_last_lr()[0]})
