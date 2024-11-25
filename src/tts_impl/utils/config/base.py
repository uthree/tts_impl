import inspect
from copy import copy as deep_copy
from dataclasses import asdict, field, make_dataclass
from functools import partial
from typing import Any, Generic, Optional, Protocol, Self, TypeVar

ConfigDataclass = TypeVar("ConfigDataclass")  # generic type


class Configuratible(Generic[ConfigDataclass], Protocol):
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


def derive_config(cls: type, cls_name: Optional[str] = None) -> type:
    """
    Derive configuration dataclass from constructor

    Args:
        cls: type
        cls_name: str, default = f"{class_name}Config"

    Returns:
        type: configuration dataclass
    """
    if cls_name is None:
        cls_name = cls.__name__ + "Config"

    signature = inspect.signature(cls.__init__)
    fields = []

    for param_name, param in signature.parameters.items():
        # Skip "self"
        if param_name == "self":
            continue

        # Handle default value
        if param.default is not inspect.Parameter.empty:
            default = field(
                default_factory=partial(lambda v: deep_copy(v), param.default)
            )
        else:
            default = field(default_factory=lambda: None)

        # Handle type annotation
        annotation = (
            param.annotation if param.annotation is not inspect.Parameter.empty else Any
        )

        # Add to fields
        fields.append((param_name, annotation, default))

    # Create dataclass
    return make_dataclass(cls_name, fields)
