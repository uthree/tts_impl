from typing import Protocol


class BuildFromConfig(Protocol):
    def build_from_config(self, config):
        pass