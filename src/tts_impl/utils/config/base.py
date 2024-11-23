from typing import Any, Mapping, Self, Generic, TypeVar, Protocol


ConfigDataclass = TypeVar('ConfigDataclass')
class Configuratible(Generic[ConfigDataclass]):
    @classmethod
    def build_from_config(cls, config: ConfigDataclass) -> Self:
        """
        Construct model from configuration dataclass
        """

    @classmethod
    def default_config(self) -> ConfigDataclass:
        """
        Get default configuration value
        """
        return ConfigDataclass()