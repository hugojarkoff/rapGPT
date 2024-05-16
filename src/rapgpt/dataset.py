import torch
from torch.utils.data import Dataset, DataLoader
from rapgpt.config import Config
from rapgpt.data import Corpus, Artist
from rapgpt.encoder import Encoder


class ArtistDataset(Dataset):
    def __init__(self, artist: Artist, encoder: Encoder) -> None:
        self.encoder = encoder
        self.artist = artist
        self.data: list[int] = self.encoder.encode_data(self.artist.lyrics)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Generate a sequence of tokens starting at the given index
        sequence = self.data[index : index + self.encoder.context_length]
        targets = sequence[1:]  # The target is the next token in the sequence
        inputs = sequence[:-1]  # The input is the current token in the sequence

        # Add padding
        inputs = self.encoder.add_padding(inputs)
        targets = self.encoder.add_padding(targets)

        # Convert everything to PyTorch tensors
        inputs = torch.tensor(inputs)
        targets = torch.tensor(targets)

        return inputs, targets


if __name__ == "__main__":
    # Example usage:
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

    for batch in dataloader:
        inputs, targets = batch
        # Now you can use inputs, targets, and mask in your training loop
        break

    print(inputs)
    print(targets)
    print(dataset.encoder.encoding.encode("<PAD>"))
