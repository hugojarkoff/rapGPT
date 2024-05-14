import tiktoken
from rapgpt.config import DatasetEncodingConfig


class Encoder:
    def __init__(self, dataset_encoding_config: DatasetEncodingConfig) -> None:
        self.encoding = tiktoken.get_encoding(dataset_encoding_config.encoding)
        self.max_length = dataset_encoding_config.max_length
        self.padding_token = dataset_encoding_config.padding_token

    def encode_data(self, data: str) -> list[int]:
        return self.encoding.encode(data)

    def decode_data(self, data: list[int]) -> str:
        return self.encoding.decode(data)

    def add_padding(self, sequence: list[int]) -> list[int]:
        if (
            len(sequence) < self.max_length
        ):  # Only pad if the sequence is shorter than the maximum length
            pad_length = self.max_length - len(sequence)
            padding = [self.encoding.encode(self.padding_token)[0]] * pad_length
            return sequence + padding
        else:
            return sequence

    @property
    def vocab_size(self) -> int:
        return self.encoding.max_token_value + 1
