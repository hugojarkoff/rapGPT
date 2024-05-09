import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from rapgpt.config import Config
from rapgpt.data import Corpus, Artist


class RapperDataset(Dataset):
    def __init__(self, artist: Artist, config: Config) -> None:
        self.config = config
        self.enc = tiktoken.get_encoding(config.dataset.encoding)
        self.max_length = config.dataset.max_length
        self.padding_token = config.dataset.padding_token

        self.artist = artist  # TODO: Find a way to encode the artist name as token
        self.data = self._encode_data(self.artist.lyrics)

    def _encode_data(self, data: str) -> list[int]:
        return self.enc.encode(data)

    def _add_padding(self, sequence: list[int]) -> list[int]:
        if (
            len(sequence) < self.max_length
        ):  # Only pad if the sequence is shorter than the maximum length
            pad_length = self.max_length - len(sequence)
            padding = [self.enc.encode(self.padding_token)[0]] * pad_length
            return sequence + padding
        else:
            return sequence

    @property
    def vocab_size(self) -> int:
        return self.enc.max_token_value + 1

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[list[int], list[int], list[int]]:
        # Generate a sequence of tokens starting at the given index
        sequence = self.data[index : index + self.max_length]
        targets = sequence[1:]  # The target is the next token in the sequence
        inputs = sequence[:-1]  # The input is the current token in the sequence

        # Add padding
        inputs = self._add_padding(inputs)
        targets = self._add_padding(targets)

        # Create a mask
        mask = [1] * len(inputs)

        # Convert everything to PyTorch tensors
        inputs = torch.tensor(inputs)
        targets = torch.tensor(targets)
        mask = torch.tensor(mask)

        return inputs, targets, mask


if __name__ == "__main__":
    # Example usage:
    config = Config()
    corpus = Corpus(config)
    dataset = RapperDataset(corpus.artists[0], config)
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
    print(dataset.enc.encode("<PAD>"))
