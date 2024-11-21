import os
from pathlib import Path
from typing import Any, Generator, List, Optional, Union

import torch
from tqdm import tqdm


class DataCollector:
    """
    Base class for collect raw data.
    """

    def __iter__(self) -> Generator[dict, None, None]:
        pass


class Extractor:
    """
    Base class for extract features from raw data.
    """

    def extract(data: dict) -> dict:
        pass

    def __call__(self, *args, **kwargs):
        return self.extract(*args, **kwargs)


class FunctionalExtractor(Extractor):
    """
    Extractor for simple simple function (e.g. MelSpectrogram, estimate_f0)
    """

    def __init__(self, input_key: str, output_key: str, fn: callable, nograd=True):
        self.input_key = input_key
        self.output_key = output_key
        self.fn = fn
        self.nograd = nograd

    def extract(self, data: dict) -> dict:
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
        self, cache_dir: Union[str, os.PathLike] = "./dataset_cache", *args, **kwargs
    ):
        self.cache_dir = Path(cache_dir)
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
    Base class of preprocessor
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
        self.collectors.append(collector)

    def with_extractor(self, extractor: Extractor):
        self.extractors.append(extractor)

    def with_writer(self, writer: CacheWriter):
        self.writer = writer

    def run(self):
        assert self.writer is not None, "CacheWriter required, but not given."

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
