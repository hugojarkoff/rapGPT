import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from rapgpt.model import TransformerModel
from rapgpt.dataset import ArtistDataset
from rapgpt.config import Config
from rapgpt.dataset import Encoder
from rapgpt.data import Corpus
from loguru import logger


class Trainer:
    def __init__(self, config: Config) -> None:
        self.config = config

        # General config
        self.encoder = Encoder(dataset_encoding_config=config.dataset_encoding)
        self.corpus = Corpus(data_path=config.data.path)
        self.artist = self.corpus.artists[0]  # TODO: Change

        ## Dataset
        self.dataset = ArtistDataset(
            artist=self.artist,
            encoder=self.encoder,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.dataloader.batch_size,
            shuffle=config.dataloader.shuffle,
        )

        ## Model
        self.device = (
            torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
        )
        vocab_size = self.encoder.vocab_size
        self.model = TransformerModel(vocab_size=vocab_size, config=config).to(
            self.device
        )

        ## Loss Function and Optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.training.lr)

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
        logger.info("Training model")
        generated_lyrics = self.generate()
        logger.info(f"Initial generated lyrics: {generated_lyrics}")

        # Training loop
        num_epochs = self.config.training.num_epochs
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0


            breaker = 0
            for inputs, targets in self.dataloader:
                inputs, targets = (
                    inputs.long().to(self.device),
                    targets.long().to(self.device),
                )

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(inputs)

                # Loss computation
                output_flat = output.reshape(-1, output.shape[2])  # Flatten output
                targets_flat = targets.reshape(-1)  # Flatten targets
                loss = self.criterion(output_flat, targets_flat)  # Compute loss
                logger.info(f"Loss: {loss}")
                epoch_loss += loss

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                breaker+=1
                if breaker==20:
                    break

            # Print the average loss for the epoch
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(self.dataloader)}"
            )
            logger.info(self.generate())
