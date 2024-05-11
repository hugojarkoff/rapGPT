import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from rapgpt.config import Config, DatasetEncodingConfig
from rapgpt.data import Corpus, Artist


class Encoder:
    def __init__(self, dataset_encoding_config: DatasetEncodingConfig) -> None:
        self.encoding = tiktoken.get_encoding(dataset_encoding_config.encoding)
        self.max_length = dataset_encoding_config.max_length
        self.padding_token = dataset_encoding_config.padding_token

    def encode_data(self, data: str) -> list[int]:
        return self.encoding.encode(data)

    def add_padding(self, sequence: list[int]) -> list[int]:
        if (
            len(sequence) < self.max_length
        ):  # Only pad if the sequence is shorter than the maximum length
            pad_length = self.max_length - len(sequence)
            padding = [self.encoding.encode(self.padding_token)[0]] * pad_length
            return sequence + padding
        else:
            return sequence


class ArtistDataset(Dataset):
    def __init__(self, artist: Artist, encoder: Encoder) -> None:
        self.encoder = encoder
        self.artist = (
            artist  # TODO: Find a way to encode the artist name as token into self.data
        )
        self.data = self.encoder.encode_data(self.artist.lyrics)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Generate a sequence of tokens starting at the given index
        sequence = self.data[index : index + encoder.max_length]
        targets = sequence[1:]  # The target is the next token in the sequence
        inputs = sequence[:-1]  # The input is the current token in the sequence

        # Add padding
        inputs = self.encoder.add_padding(inputs)
        targets = self.encoder.add_padding(targets)

        # Create a mask
        mask = [1] * len(inputs)

        # Convert everything to PyTorch tensors
        inputs = torch.tensor(inputs)
        targets = torch.tensor(targets)
        mask = torch.tensor(mask)

        return inputs, targets, mask

    @property
    def vocab_size(self) -> int:
        return self.encoder.encoding.max_token_value + 1


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
        inputs, targets, mask = batch
        # Now you can use inputs, targets, and mask in your training loop
        break

    print(inputs)
    print(targets)
    print(dataset.encoder.encoding.encode("<PAD>"))
