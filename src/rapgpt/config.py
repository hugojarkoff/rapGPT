from typing import Tuple, Type
from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class DataConfig(BaseModel):
    path: str = "data/french_rap_lyrics"


class DatasetEncodingConfig(BaseModel):
    max_length: int = 512  # Max context size, padded if shorter
    encoding: str = "r50k_base"
    padding_token: str = "<PAD>"


class DataloaderConfig(BaseModel):
    shuffle: bool = True
    batch_size: int = 2


class ModelConfig(BaseModel):
    num_heads: int = 4
    hidden_dim: int = 128
    num_layers: int = 3


class TrainingConfig(BaseModel):
    lr: float = 1e-3
    num_epochs: int = 10


class Config(BaseSettings):
    revision: str = "main"
    data: DataConfig = DataConfig()
    dataset_encoding: DatasetEncodingConfig = DatasetEncodingConfig()
    dataloader: DataloaderConfig = DataloaderConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()

    model_config = SettingsConfigDict(toml_file="configs/config.toml")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls),)
