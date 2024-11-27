import argparse
import inspect
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import yaml
from lightning import LightningModule, Trainer, LightningDataModule
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from rich import print
from rich.table import Column, Table
from rich_argparse import RichHelpFormatter
from omegaconf import OmegaConf


def build_argparser_for_fn(fn: function):
    """
    Generate argument parser automatically
    """
    parser = argparse.ArgumentParser(
        description=fn.__doc__, formatter_class=RichHelpFormatter
    )
    sig = inspect.signature(fn)
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


default_trainer_config = {
    "devices": "auto",
    "max_epochs": None,
    "max_steps": None,
    "precision": "32",
    "log_every_n_steps": 50
}

class Recipe:
    """
    Recipe(WIP) experimental feature
    """

    def __init__(self, target_module: LightningModule, datamodule: LightningDataModule):
        pass