from typing import Any, Mapping, Self


class BuildFromConfig:
    def build_from_config(cls, config: Mapping[str, Any]) -> Self:
        """
        Construct model from configuration dataclass
        """
