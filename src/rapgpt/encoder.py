import tiktoken
from rapgpt.config import Config


class Encoder:
    def __init__(self, config: Config) -> None:
        self.encoding = tiktoken.get_encoding(config.corpus.encoding)

    def encode_data(self, data: str) -> list[int]:
        return self.encoding.encode(data)

    def decode_data(self, data: list[int]) -> str:
        return self.encoding.decode(data)

    @property
    def vocab_size(self) -> int:
        return self.encoding.max_token_value + 1
