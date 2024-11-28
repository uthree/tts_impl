import argparse
import inspect
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import torch
import yaml
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import RichProgressBar
from rich import print
from rich_argparse import RichHelpFormatter
from tts_impl.utils.config import arguments_dataclass_of


def build_argparser_for_fn(fn: callable):
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


class ManualSave(Callback):
    def __init__(self, path: Path):
        self.path = path

    def on_train_epoch_end(self, trainer, pl_module: LightningModule, *args, **kwargs):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pl_module.state_dict(), self.path)


class Recipe:
    """
    Recipe(WIP) experimental feature
    """

    def __init__(self, target_module: LightningModule, name: Optional[str] = None):
        self.TargetModule = target_module

        if name is None:
            self.name = target_module.__name__
        else:
            self.name = name

        self.config_root_dir = Path("config") / self.name
        self.ckpt_root_dir = Path("checkpoint") / self.name

        self.ckpt_name = "model"

    def checkopint_callback(self) -> ManualSave:
        return ManualSave(self.ckpt_root_dir / (self.ckpt_name + ".pt"))

    def prepare_trainer(
        self, epochs: int = 100, precision: str = "bf16-mixed"
    ) -> Trainer:
        # initialize trainer
        trainer = Trainer(
            max_epochs=epochs,
            precision=precision,
            callbacks=[RichProgressBar(), self.checkopint_callback()],
            log_every_n_steps=50,
        )

        return trainer

    def prepare_datamodule(self) -> LightningDataModule:
        raise NotImplemented("prepare_datamodule is not implemented!!")

    def load_config(self, config_name: str = "default"):
        path = self.config_root_dir / (config_name + ".yml")
        with open(path) as f:
            return yaml.safe_load(f.read())

    def train(self, config_name: str = "default"):
        self.ckpt_name = config_name
        config = self.load_config(config_name)
        datamodule = self.prepare_datamodule(**config["datamodule"])
        trainer = self.prepare_trainer(**config["trainer"])
        ckpt_path = self.ckpt_root_dir / (config_name + ".pt")
        if ckpt_path.exists():
            print(f"Ckeckpoint {ckpt_path} found, loading ckeckpoint")
            model = self.TargetModule(**config["model"])
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        else:
            print(f"Ckeckpoint {ckpt_path} is not found. initializing model")
            model = self.TargetModule(**config["model"])
        trainer.fit(model, datamodule)
        print("Training Complete!")

    def preprocess(self, **config):
        raise NotImplemented("preprocess is not implemented!!")

    def prepare_config_dir(self):
        config = self.TargetModule.default_config()
        model_config_dict = asdict(config)
        datamodule_config_cls = arguments_dataclass_of(
            getattr(self, "prepare_datamodule")
        )
        datamodule_config_dict = asdict(datamodule_config_cls())
        trainer_config_cls = arguments_dataclass_of(getattr(self, "prepare_trainer"))
        trainer_config_dict = asdict(trainer_config_cls())
        preprocess_config_cls = arguments_dataclass_of(getattr(self, "preprocess"))
        preprocess_config_dict = asdict(preprocess_config_cls())

        config_dict = {
            "model": model_config_dict,
            "datamodule": datamodule_config_dict,
            "trainer": trainer_config_dict,
            "preprocess": preprocess_config_dict,
        }

        self.config_root_dir.mkdir(parents=True, exist_ok=True)
        default_config_path = self.config_root_dir / "default.yml"
        if not default_config_path.exists():
            with open(default_config_path, mode="w") as f:
                yaml.dump(config_dict, f)

    def cli(self):
        """
        execute as the command line interface
        """
        if self is self.__class__:
            self.__class__().cli()

        parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter)
        parser.add_argument("command", choices=["preprocess", "train", "setup"])
        parser.add_argument("-c", "--config", default="default")

        args = parser.parse_args()
        if args.command == "train":
            self.train(args.config)
        elif args.command == "preprocess":
            cfg = self.load_config(args.config)
            self.preprocess(**cfg["preprocess"])
        elif args.command == "setup":
            self.prepare_config_dir()
