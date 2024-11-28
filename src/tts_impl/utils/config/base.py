import inspect
from copy import copy as deep_copy
from dataclasses import field, make_dataclass
from functools import partial
from typing import Any, Optional

from .arguments import arguments_dataclass_of

"""
激ヤバ黒魔術コード！！！！
using DARK-SIDE POWER !!!!
"""


def keys(self):
    return self.__dict__.keys()


def items(self):
    return self.__dict__.items()


def values(self):
    return self.__dict__.values()


def __len__(self):
    return len(self.__dict__)


def __getitem__(self, key):
    return getattr(self, key)


def __setitem__(self, key, value):
    return setattr(self, key, value)


def __contains__(self, key):
    return key in self.__dict__


def __repr__(self):
    return self.__dict__.__repr__()


def config_of(cls: type, cls_name: Optional[str] = None) -> type:
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

    config_cls = arguments_dataclass_of(cls.__init__, cls_name)

    # Implementation
    config_cls.keys = keys
    config_cls.items = items
    config_cls.values = values
    config_cls.__getitem__ = __getitem__
    config_cls.__setitem__ = __setitem__
    config_cls.__contains__ = __contains__
    config_cls.__repr__ = __repr__

    return config_cls


@classmethod
def default_config(cls):
    return cls.Config()


def derive_config(cls, cls_name: Optional[None] = None):
    """
    Decorator for deriving configuration dataclass and attach it to Class.Config automatically.
    """
    # Generate the configuration class
    config_cls = config_of(cls, cls_name=cls_name)

    # Attach it to the original class
    cls.Config = config_cls

    cls.default_config = default_config
    return cls
