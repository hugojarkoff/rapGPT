import tiktoken
from rapgpt.config import DatasetEncodingConfig


class Encoder:
    def __init__(self, dataset_encoding_config: DatasetEncodingConfig) -> None:
        self.encoding = tiktoken.get_encoding(dataset_encoding_config.encoding)
        self.context_length = dataset_encoding_config.context_length

    def encode_data(self, data: str) -> list[int]:
        return self.encoding.encode(data)

    def decode_data(self, data: list[int]) -> str:
        return self.encoding.decode(data)

    @property
    def vocab_size(self) -> int:
        return self.encoding.max_token_value + 1
