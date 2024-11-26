import argparse
import inspect
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import yaml
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from rich import print
from rich.table import Column, Table
from rich_argparse import RichHelpFormatter


# generate argument parser automatically
def build_argparser_for_fn(func: function):
    """
    Generate argument parser automatically
    """
    parser = argparse.ArgumentParser(
        description=func.__doc__, formatter_class=RichHelpFormatter
    )
    sig = inspect.signature(func)
    for name, param in sig.parameters.items():
        arg_type = (
            param.annotation if param.annotation != inspect.Parameter.empty else str
        )
        default = param.default
        if default == inspect.Parameter.empty:
            # required arg.
            parser.add_argument(f"{name}", type=arg_type, help=f"{name} (required)")
        else:
            # optional arg.
            parser.add_argument(
                f"--{name}",
                type=arg_type,
                default=default,
                help=f"{name} (default: {default})",
            )
    return parser


class Recipe:
    """
    Recipe(WIP)
    """

    def __init__(self, name: str = None):
        if name is None:
            name = self.__class__.__name__
        self.name = name

        self.models = {}
        self.preprocess_functions = {}

        self.config_root_dir = Path("config") / self.name
        self.weight_root_dir = Path("weight") / self.name

    def register_model(self, model_cls: LightningModule, name="model"):
        self.models[name] = model_cls

    def _register_preprocess_fn(self, fn: function, name="preprocess"):
        self.preprocess_functions[name] = fn

    def preprocess(self, fn: function):
        """
        decorator
        """
        self._register_preprocess_fn(fn, name=fn.__name__)
        return fn

    def prepare_trainer(self, ckpt_path: Path, epoch: int = 1):
        callbacks = [
            RichProgressBar(),
            ModelCheckpoint(dirpath=ckpt_path.parent, filename=ckpt_path.name),
        ]
        trainer = Trainer(callbacks=callbacks, max_epochs=epoch)
        return trainer

    def prepare_default_configs(self):
        for model_name, model_cls in self.models.items():
            model_config_dir = self.config_root_dir / model_name
            model_config_dir.mkdir(parents=True, exist_ok=True)
            model_default_config_path: Path = self._generate_config_path(
                model_name, "default"
            )
            # Initialize default config file if not exists
            if not model_default_config_path.exists():
                model_default_config = asdict(model_cls.default_config())
                with open(model_default_config_path, mode="w") as f:
                    yaml.dump(model_default_config, f)

    def _generate_config_path(
        self, model_name: str = "model", config_name: str = "default"
    ) -> Path:
        return self.config_root_dir / model_name / config_name + ".yaml"

    def _generate_ckpt_path(
        self, model_name: str = "model", ckpt_name: str = "checkpoint"
    ) -> Path:
        return self.weight_root_dir / model_name / ckpt_name + ".ckpt"

    def load_model(self, model_name: str = "model", ckpt_name: str = "checkpoint"):
        model_weight_path: Path = self._generate_ckpt_path(model_name, ckpt_name)
        model_cls: LightningModule = self.models[model_name]
        model = model_cls.load_from_checkpoint(model_weight_path)
        return model

    def train(self, model_name: str = "model", ckpt_name: str = "checkpoint"):
        pass

    def cli(self, args):
        """
        Execute CLI
        """
        pass
