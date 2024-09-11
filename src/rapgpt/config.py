import tomli
from typing import Type, TypeVar
from pydantic import BaseModel, ConfigDict
from pathlib import Path


class DataConfig(BaseModel):
    path: str = "data/french_rap_lyrics"


class DatasetEncodingConfig(BaseModel):
    context_length: int = 512
    encoding: str = "r50k_base"


class ModelConfig(BaseModel):
    num_heads: int = 4
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1


class TrainingConfig(BaseModel):
    lr: float = 1e-3
    num_steps: int = 10000
    num_steps_val: int = 250
    evaluation_cycle: int = 150
    batch_size: int = 16


class EvalConfig(BaseModel):
    sample_text: str = "Les vrais savent"
    new_tokens: int = 30
    artist_token: int = 0


class WandbConfig(BaseModel):
    project: str = "rapGPT"
    mode: str = "online"  # offline or online or disabled
    group: str = "dev"
    tags: list[str] = ["dev", "debug"]


T = TypeVar("T", bound="Config")


class Config(BaseModel):
    revision: str = "main"
    data: DataConfig = DataConfig()
    dataset_encoding: DatasetEncodingConfig = DatasetEncodingConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvalConfig = EvalConfig()
    wandb: WandbConfig = WandbConfig()

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def load_from_toml(cls: Type[T], toml_path: Path | str) -> T:
        with open(file=toml_path, mode="rb") as f:
            config_dict = tomli.load(f)

        return cls(**config_dict)
