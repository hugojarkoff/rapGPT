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
        self.model = TransformerModel(
            input_dim=vocab_size,
            num_heads=self.config.model.num_heads,
            hidden_dim=self.config.model.hidden_dim,
            num_layers=self.config.model.num_layers,
        ).to(self.device)

        ## Loss Function and Optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.training.lr)

    @torch.no_grad()
    def generate(
        self, max_length: int = 100, sample_text: str = "Les vrais savent"
    ) -> str:
        self.model.eval()

        # Encode the seed text
        sample_input = self.encoder.encode_data(sample_text)
        sample_input = torch.tensor(sample_input).unsqueeze(0).to(self.device)

        # Generate new tokens
        for _ in range(max_length):
            mask = torch.ones_like(sample_input)
            output = self.model(sample_input, mask=mask)

            # Sample the next token
            output = output[0, -1, :]  # Take the last token
            output = torch.softmax(output, dim=-1)  # Apply softmax
            next_token = torch.multinomial(output, 1)  # Sample from the distribution

            # Append the next token to the seed
            sample_input = torch.cat([sample_input, next_token.unsqueeze(0)], dim=1)

        # Decode the generated tokens
        return self.encoder.decode_data(sample_input[0].tolist())

    def train(
        self,
    ) -> None:
        logger.info("Training model")
        logger.info("Generating lyrics: ", self.generate())

        # Training loop
        num_epochs = self.config.training.num_epochs
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0

            for inputs, targets, mask in self.dataloader:
                inputs, targets, mask = (
                    inputs.long().to(self.device),
                    targets.long().to(self.device),
                    mask.long().to(self.device),
                )

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(inputs, mask=mask)

                # Loss computation
                output_flat = output.reshape(-1, output.shape[2])  # Flatten output
                targets_flat = targets.reshape(-1)  # Flatten targets
                loss = self.criterion(output_flat, targets_flat)  # Compute loss
                total_loss += loss

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

            # Print the average loss for the epoch
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(self.dataloader)}"
            )
            logger.info(self.generate())
