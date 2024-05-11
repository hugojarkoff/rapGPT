import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from rapgpt.model import TransformerModel
from rapgpt.dataset import ArtistDataset
from rapgpt.config import Config
from rapgpt.dataset import Encoder
from rapgpt.data import Corpus

# Define your dataset and dataloader
config = Config()
encoder = Encoder(dataset_encoding_config=config.dataset_encoding)
corpus = Corpus(data_path=config.data.path)
artist = corpus.artists[0]
dataset = ArtistDataset(
    artist=artist,
    encoder=encoder,
)
dataloader = DataLoader(
    dataset,
    batch_size=config.dataloader.batch_size,
    shuffle=config.dataloader.shuffle,
)


# Initialize your model
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
vocab_size = dataset.vocab_size

model = TransformerModel(
    input_dim=vocab_size, num_heads=4, hidden_dim=128, num_layers=3
).to(device)

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def generate(model, max_length: int = 100):
    model.eval()

    seed_text = "Les vrais savent"

    with torch.no_grad():
        # Encode the seed text
        seed = dataset.encoder.encode_data(seed_text)
        seed = torch.tensor(seed).unsqueeze(0).to(device)

        # Generate new tokens
        for _ in range(max_length):
            mask = torch.ones_like(seed)
            output = model(seed, mask=mask)

            # Sample the next token
            output = output[0, -1, :]  # Take the last token
            output = torch.softmax(output, dim=-1)  # Apply softmax
            next_token = torch.multinomial(output, 1)  # Sample from the distribution

            # Append the next token to the seed
            seed = torch.cat([seed, next_token.unsqueeze(0)], dim=1)

        # Decode the generated tokens
        return dataset.encoder.encoding.decode(seed[0].tolist())


def train():
    print(generate(model))

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for inputs, targets, mask in dataloader:
            inputs, targets, mask = (
                inputs.long().to(device),
                targets.long().to(device),
                mask.long().to(device),
            )

            # Forward pass
            optimizer.zero_grad()
            output = model(inputs, mask=mask)

            # Loss computation
            output_flat = output.reshape(-1, output.shape[2])  # Flatten output
            targets_flat = targets.reshape(-1)  # Flatten targets
            loss = criterion(output_flat, targets_flat)  # Compute loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Print the average loss for the epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")
        print(generate(model))


if __name__ == "__main__":
    train()
    print(generate(model))
