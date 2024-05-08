from typing import Tuple, Type
from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class DataConfig(BaseModel):
    path: str


class DatasetConfig(BaseModel):
    max_length: int
    encoding: str


class DataloaderConfig(BaseModel):
    shuffle: bool
    batch_size: int


class Config(BaseSettings):
    revision: str
    data: DataConfig
    dataset: DatasetConfig
    dataloader: DataloaderConfig

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
