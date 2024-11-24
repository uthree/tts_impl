from dataclasses import asdict
from typing import Any, Generic, Mapping, Protocol, Self, TypeVar

ConfigDataclass = TypeVar("ConfigDataclass")


class Configuratible(Generic[ConfigDataclass]):
    @classmethod
    def build_from_config(cls, config: ConfigDataclass) -> Self:
        """
        Construct model from configuration dataclass
        """
        return cls.__init__(**asdict(config))

    @classmethod
    def default_config(self) -> ConfigDataclass:
        """
        Get default configuration value
        """
        return ConfigDataclass()
