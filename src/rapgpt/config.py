import tomli
from typing import Type, TypeVar
from pydantic import BaseModel, ConfigDict
from pathlib import Path


class DataConfig(BaseModel):
    path: str = "data/french_rap_lyrics"


class CorpusConfig(BaseModel):
    seed: int = 42
    context_length: int = 512
    encoding: str = "r50k_base"
    split_train_val: float = 0.8
    # Minimum number of tokens in artist lyrics, else discarded from Corpus
    min_artist_length: int = 50000


class ModelConfig(BaseModel):
    num_heads: int = 4
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1


class TrainingConfig(BaseModel):
    lr: float = 1e-3
    seed: int = 42
    batch_size: int = 16
    # Total training steps (no epochs in this project)
    num_training_steps: int = 10000
    # Num validation steps per evaluation cycle
    num_validation_steps: int = 250
    # Launch evaluation cycle every evaluation_cycle steps
    evaluation_cycle: int = 150


class ExpSchedulerConfig(BaseModel):
    gamma: float = 0.9


class EvalConfig(BaseModel):
    sample_text: str = "Les vrais savent"
    new_tokens: int = 30
    artist_token: int = 0


class WandbConfig(BaseModel):
    project: str = "rapGPT"
    mode: str = "online"  # offline or online or disabled
    group: str = "dev"
    tags: list[str] = ["dev", "debug"]

class CheckpointCallbackConfig(BaseModel):
    save_path: str = "checkpoints"


T = TypeVar("T", bound="Config")


class Config(BaseModel):
    revision: str = "main"
    data: DataConfig = DataConfig()
    corpus: CorpusConfig = CorpusConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvalConfig = EvalConfig()
    scheduler: ExpSchedulerConfig = ExpSchedulerConfig()
    wandb: WandbConfig = WandbConfig()
    checkpoint: CheckpointCallbackConfig = CheckpointCallbackConfig()

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def load_from_toml(cls: Type[T], toml_path: Path | str) -> T:
        with open(file=toml_path, mode="rb") as f:
            config_dict = tomli.load(f)

        return cls(**config_dict)
