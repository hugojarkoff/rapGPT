import torch
import torch.nn as nn
import torch.optim as optim
from rapgpt.model import TransformerModel
from rapgpt.config import Config
from rapgpt.encoder import Encoder
from rapgpt.data import Corpus
from loguru import logger
import wandb

loggable = str | int | float


class Trainer:
    def __init__(self, config: Config) -> None:
        self.config = config
        wandb.init(
            project=config.wandb.project,
            mode=config.wandb.mode,
            tags=config.wandb.tags,
            group=config.wandb.group,
        )

        ## Dataset
        self.encoder = Encoder(dataset_encoding_config=config.dataset_encoding)
        self.corpus = Corpus(
            data_path=config.data.path,
            encoder=self.encoder
        )

        ## Model
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        vocab_size = self.encoder.vocab_size
        self.model = TransformerModel(
            vocab_size=vocab_size, 
            artists_size=len(self.corpus.artist_encoding),
            config=config,
        ).to(
            device=self.device
        )

        ## Loss Function and Optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.training.lr)

    def log(self, message: str | dict[str, loggable]) -> None:
        if isinstance(message, str):
            logger.info(message)
            return
        assert isinstance(message, dict)
        for k, v in message.items():
            logger.info(f"{k}:{v}")
        wandb.log(message)

    @torch.no_grad()
    def generate(
        self, sample_text: str = "Les vrais savent", new_tokens: int = 30
    ) -> str:
        self.model.eval()

        # Encode the seed text
        sample_input = self.encoder.encode_data(sample_text)
        sample_input = torch.tensor(sample_input).unsqueeze(0).to(self.device)

        output = self.model.generate(
            sample_input,
            new_tokens,
        )

        # Decode the generated tokens
        return self.encoder.decode_data(output[0].tolist())

    def train(
        self,
    ) -> None:
        self.log("Training model")
        generated_lyrics = self.generate(
            sample_text=self.config.evaluation.sample_text,
            new_tokens=self.config.evaluation.new_tokens,
        )
        self.log({"Eval lyrics": generated_lyrics})

        # Training loop
        for step in range(self.config.training.num_steps):
            self.model.train()

            inputs, targets, artists = self.corpus.get_random_batch(
                batch_size=self.config.training.batch_size,
                block_size=self.config.dataset_encoding.context_length
            )

            inputs, targets, artists = (
                inputs.long().to(self.device),
                targets.long().to(self.device),
                artists.long().to(self.device)
            )

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(inputs, artists)

            # Loss computation
            output_flat = output.reshape(-1, output.shape[2])  # Flatten output
            targets_flat = targets.reshape(-1)  # Flatten targets
            loss = self.loss_fn(output_flat, targets_flat)  # Compute loss

            self.log({"Batch loss": loss})

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if step % 250 == 0:

                # Evaluation loop every 250 steps
                generated_lyrics = self.generate(
                    sample_text=self.config.evaluation.sample_text,
                    new_tokens=self.config.evaluation.new_tokens,
                )
                self.log({"Eval lyrics": generated_lyrics})
