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


class DatasetConfig(BaseModel):
    max_length: int = 512
    encoding: str = "r50k_base"
    padding_token: str = "<PAD>"


class DataloaderConfig(BaseModel):
    shuffle: bool = True
    batch_size: int = 2


class Config(BaseSettings):
    revision: str = "main"
    data: DataConfig = DataConfig()
    dataset: DatasetConfig = DatasetConfig()
    dataloader: DataloaderConfig = DataloaderConfig()

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


def unit_test():
    settings = Config()
    print(settings)


if __name__ == "__main__":
    # TODO: Move/refactor later
    unit_test()
