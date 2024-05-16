import tomli
from typing import Type, TypeVar
from pydantic import BaseModel, ConfigDict
from pathlib import Path


class DataConfig(BaseModel):
    path: str = "data/french_rap_lyrics"


class DatasetEncodingConfig(BaseModel):
    context_length: int = 512  # Max context size, padded if shorter
    encoding: str = "r50k_base"
    padding_token: str = "<PAD>"


class DataloaderConfig(BaseModel):
    shuffle: bool = True
    batch_size: int = 16


class ModelConfig(BaseModel):
    num_heads: int = 4
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1


class TrainingConfig(BaseModel):
    lr: float = 1e-3
    num_epochs: int = 10


class EvalConfig(BaseModel):
    sample_text: str = "Les vrais savent"
    new_tokens: int = 30


T = TypeVar("T", bound="Config")


class Config(BaseModel):
    revision: str = "main"
    data: DataConfig = DataConfig()
    dataset_encoding: DatasetEncodingConfig = DatasetEncodingConfig()
    dataloader: DataloaderConfig = DataloaderConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvalConfig = EvalConfig()

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def load_from_toml(cls: Type[T], toml_path: Path | str) -> T:
        with open(file=toml_path, mode="rb") as f:
            config_dict = tomli.load(f)

        return cls(**config_dict)
