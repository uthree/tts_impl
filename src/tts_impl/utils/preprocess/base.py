import logging
import os
import shutil
from collections.abc import Generator
from logging import Logger
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.logging import RichHandler


class DataCollector:
    """
    Base class for collecting raw data.
    """

    def __iter__(self) -> Generator[dict]:
        pass

    def _prepare_logger(self, console: Console, logger: Logger):
        self.console = console
        self.logger = logger

    def finalize(self):
        """
        Implement any processing you want to perform after preprocessing is complete.
        """
        pass

    def prepare(self):
        """
        This method is called only once before preprocessing is performed.
        Implement this if preparation processing is required.
        """
        pass


class Extractor:
    """
    Base class for extracting features from raw data.
    """

    def extract(data: dict[str, Any]) -> dict[str, Any]:
        """
        Processing such as feature extraction.
        """
        return data

    def _prepare_logger(self, console: Console, logger: Logger):
        self.console = console
        self.logger = logger

    def prepare(self):
        """
        This method is called only once before preprocessing is performed.
        Implement this if preparation processing is required.
        """
        pass

    def finalize(self):
        """
        Implement any processing you want to perform after preprocessing is complete.
        """
        pass

    def __call__(self, data):
        """
        Alias.
        Same as the method `extract`.
        """
        return self.extract(data)


class CombinedExtractor(Extractor):
    def __init__(self, *extractors: list[Extractor]):
        super().__init__()
        self.extractors = extractors

    def prepare(self):
        for e in self.extractors:
            e.prepare()

    def finalize(self):
        for e in self.extractors:
            e.finalize()

    def extract(self, data: dict[str, Any]) -> dict[str, Any]:
        for e in self.extractors:
            data = e.extract(data)
        return data


class FunctionalExtractor(Extractor):
    """
    Extractor for simple simple function.
    """

    def __init__(self, input_key: str, output_key: str, fn: callable, nograd=True):
        """
        Args:
            input_key: str, The key to be processed
            output_key: str, Key to save the processing result
            fn: callable, A function that performs processing. The first argument is the input_key data.
            nograd: bool, If True, gradients are not calculated. Preprocessing usually does not require gradients, so the default is True.
        """
        self.input_key = input_key
        self.output_key = output_key
        self.fn = fn
        self.nograd = nograd

    def extract(self, data: dict[str, Any]) -> dict[str, Any]:
        target_data = data[self.input_key]
        if self.nograd:
            with torch.no_grad():
                output = self.fn(target_data)
        else:
            output = self.fn(target_data)
        data[self.output_key] = output
        return data


class CacheWriter:
    """
    Base class for write data to cache
    """

    def __init__(
        self,
        root: str | os.PathLike = "dataset_cache",
        delete_old_cache=True,
        *args,
        **kwargs,
    ):
        """
        Args
            root: str | os.PathLike, Directory for caching datasets.
            remove_old_cache: bool
        """
        self.delete_old_cache = delete_old_cache
        self.root = Path(root)

    def _prepare_logger(self, console: Console, logger: Logger):
        self.console = console
        self.logger = logger

    def prepare(self):
        """
        This method is called only once before preprocessing is performed.
        Implement this if preparation processing is required.

        The default implementation is to delete old caches and create a directory for the cache.
        """
        if self.delete_old_cache:
            if self.root.exists():
                shutil.rmtree(self.root)
                self.logger.log(logging.INFO, f"Deleted cache directory: {self.root}")

        self.root = Path(self.root)
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)
        self.counter = 0

    def finalize(self):
        """
        Implement any processing you want to perform after preprocessing is complete, such as obtaining a list of speakers.
        """
        pass

    def write(self, data: dict[str, Any]):
        """
        The process when writing out one piece of data.
        """
        torch.save(data, self.root / f"{self.counter}.pt")
        self.counter += 1
        pass


class Preprocessor:
    """
    A class that performs preprocessing collectively.
    """

    def __init__(
        self,
        collectors: list[DataCollector] = None,
        extractors: list[Extractor] = None,
        writer: CacheWriter | None = None,
        console: Console | None = None,
        logger: Logger | None = None,
    ):
        if extractors is None:
            extractors = []
        if collectors is None:
            collectors = []
        self.collectors = collectors
        self.extractors = extractors
        self.writer = writer

        self.console = console
        self.logger = logger

    def _prepare_logger(self, level=logging.INFO, logger_name: str = "preprocess"):
        if self.logger is None:
            self.logger = logging.getLogger(logger_name)
        if self.console is None:
            self.console = Console()

        handler = RichHandler(console=self.console)
        logging.basicConfig(level=level, handlers=[handler])

        self.writer._prepare_logger(self.console, self.logger)

        for collector in self.collectors:
            collector._prepare_logger(self.console, self.logger)

        for extractor in self.extractors:
            extractor._prepare_logger(self.console, self.logger)

    def with_collector(self, collector: DataCollector):
        """
        Add collector.

        Args:
            collector: DataCollector
        """
        self.collectors.append(collector)

    def with_extractor(self, extractor: Extractor):
        """
        Add extractor.

        Args:
            extractor: Extractor
        """
        self.extractors.append(extractor)

    def with_writer(self, writer: CacheWriter):
        """
        Set cache writer.

        Args:
            writer: CacheWriter
        """
        self.writer = writer

    def run(self):
        """
        Execute preprocessing.
        """
        assert self.writer is not None, "CacheWriter required, but not given."

        self._prepare_logger()

        self.logger.debug("Preparing submodules ...")
        for e in self.extractors:
            self.logger.debug(f"Preparing {e.__class__.__name__} ...")
            e.prepare()
        self.logger.debug(f"Preparing {self.writer.__class__.__name__} ...")
        self.writer.prepare()

        self.logger.info("Start preprocessing ...")
        data_count = 0
        for collector in self.collectors:
            self.logger.debug(f"Preparing {collector.__class__.__name__} ...")
            collector.prepare()
            # start yield loop
            collector_count = 0
            for data in collector:
                for ext in self.extractors:
                    data = ext.extract(data)
                # write cache
                self.writer.write(data)
                data_count += 1
                collector_count += 1
            self.logger.log(
                logging.INFO,
                f"Processed {collector_count} item(s) in {collector.__class__.__name__}",
            )
            self.logger.debug(f"Finalizing {collector.__class__.__name__} ...")
            collector.finalize()
        self.logger.log(logging.INFO, f"Processed total {data_count} item(s).")

        self.logger.debug("Finalizing extractors...")
        for e in self.extractors:
            self.logger.debug(f"Finalizing {e.__class__.__name__} ...")
            e.finalize()
        self.logger.debug(f"Finalizing {self.writer.__class__.__name__} ...")
        self.writer.finalize()
        self.logger.log(logging.INFO, "Preprocessing complete!")
