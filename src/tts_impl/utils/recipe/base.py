import argparse
import inspect
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from omegaconf import OmegaConf
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
        self.model = None

        self._prepare_parsers()

    def _prepare_parsers(self):
        self.argparsers = {}
        self.argparsers["train"] = build_argparser_for_fn(self.train)
        self.argparsers["preprocess"] = build_argparser_for_fn(self.preprocess)
        self.argparsers["infer"] = build_argparser_for_fn(self.infer)

    def checkopint_callback(self) -> ModelCheckpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.ckpt_root_dir,
            filename=self.ckpt_name,
            enable_version_counter=False,
        )
        return checkpoint_callback

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
        return OmegaConf.load(path)

    def _indeterministic_mode(self):
        if torch.cuda.is_available():
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("medium")

    def load_model(self, config_name: str = "default"):
        model_config = self.load_config(config_name=config_name).model
        ckpt_path = self.ckpt_root_dir / f"{self.ckpt_name}.ckpt"
        if ckpt_path.exists():
            print(f"Ckeckpoint {ckpt_path} found, loading ckeckpoint")
            model = self.TargetModule.load_from_checkpoint(
                ckpt_path, map_location="cpu"
            )
        else:
            print(f"Ckeckpoint {ckpt_path} is not found. initializing model")
            model = self.TargetModule(**model_config)
        self.model = model

    def train(self, config_name: str = "default"):
        self._indeterministic_mode()
        config = self.load_config(config_name)
        datamodule = self.prepare_datamodule(**config["datamodule"])
        trainer = self.prepare_trainer(**config["trainer"])
        trainer.fit(self.model, datamodule)
        print("Training Complete!")

    def preprocess(self):
        raise NotImplemented("preprocess is not implemented!!")

    def infer(self):
        raise NotImplemented("infer is not implemented!!")

    def prepare_config_dir(self, config_name):
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
        infer_config_cls = arguments_dataclass_of(getattr(self, "infer"))
        infer_config_dict = asdict(infer_config_cls())

        config_dict = {
            "model": model_config_dict,
            "datamodule": datamodule_config_dict,
            "trainer": trainer_config_dict,
            "preprocess": preprocess_config_dict,
            "infer": infer_config_dict,
        }

        self.config_root_dir.mkdir(parents=True, exist_ok=True)
        default_config_path = self.config_root_dir / (config_name + ".yml")
        if not default_config_path.exists():
            OmegaConf.save(config_dict, default_config_path)

    def cli(self):
        """
        execute CLI
        """
        if self is self.__class__:
            self.__class__().cli()
        self.execute_command(sys.argv[1:])

    def execute_command(self, cli_args):
        parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter)
        parser.add_argument(
            "command", choices=["preprocess", "train", "setup", "infer", "exit"]
        )
        parser.add_argument("-c", "--config", default="default")
        args, remaining_argv = parser.parse_known_args(cli_args)
        if args.command == "train":
            self.ckpt_name = args.config
            self.load_model(config_name=args.config)
            self.train(config_name=args.config)
        elif args.command == "preprocess":
            self.ckpt_name = args.config
            cfg = self.load_config(config_name=args.config)
            self.preprocess(**(dict(cfg["preprocess"])))
        elif args.command == "infer":
            self.ckpt_name = args.config
            self.load_model(config_name=args.config)
            cfg = self.load_config(config_name=args.config)
            additional_args = vars(self.argparsers["infer"].parse_args(remaining_argv))
            self.infer(**(dict(cfg["infer"]) | additional_args))
        elif args.command == "setup":
            print("Setting up configuration dir ...")
            self.prepare_config_dir(args.config)
            print("Complete.")
        elif args.command == "exit":
            exit()
