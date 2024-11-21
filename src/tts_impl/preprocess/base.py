import os
import shutil
from pathlib import Path
from typing import Any, Generator, List, Optional, Union

import torch
from tqdm import tqdm


class DataCollector:
    """
    Base class for collecting raw data.
    """

    def __iter__(self) -> Generator[dict, None, None]:
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

    def prepare(self):
        """
        This method is called only once before preprocessing is performed.
        Implement this if preparation processing is required.
        """
        pass

    def __call__(self, *args, **kwargs):
        """
        Alias.
        Same as the method `extract`.
        """
        return self.extract(*args, **kwargs)


class FunctionalExtractor(Extractor):
    """
    Extractor for simple simple function (e.g. MelSpectrogram, estimate_f0)
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
        cache_dir: Union[str, os.PathLike] = "dataset_cache",
        remove_old_cache=True,
        *args,
        **kwargs,
    ):
        """
        Args
            cache_dir: Union[str, os.PathLike], Directory for caching datasets.
            remove_old_cache: bool
        """
        self.remove_old_cache = remove_old_cache
        self.cache_dir = cache_dir

    def prepare(self):
        """
        This method is called only once before preprocessing is performed.
        Implement this if preparation processing is required.

        The default implementation is to delete old caches and create a directory for the cache.
        """
        if self.remove_old_cache:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)

        self.cache_dir = Path(self.cache_dir)
        if not self.cache_dir.exists():
            self.cache_dir.mkdir()
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
        torch.save(data, self.cache_dir / f"{self.counter}.pt")
        self.counter += 1
        pass


class Preprocessor:
    """
    A class that performs preprocessing collectively.
    """

    def __init__(
        self,
        collectors: List[DataCollector] = [],
        extractors: List[Extractor] = [],
        writer: Optional[CacheWriter] = None,
    ):
        self.collectors = collectors
        self.extractors = extractors
        self.writer = writer

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

        tqdm.write("Preparing submodules...")
        for c in self.collectors:
            c.prepare()
        for e in self.extractors:
            e.prepare()
        self.writer.prepare()

        tqdm.write("Start preprocessing...")
        for collector in self.collectors:
            # start yield loop
            for data in tqdm(collector):
                for ext in self.extractors:
                    data = ext(data)
                # write cache
                self.writer.write(data)
        tqdm.write("Finalizing...")
        self.writer.finalize()
        tqdm.write("Complete!")
