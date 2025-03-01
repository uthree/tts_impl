from typing import List

from .base import LanguageModule

try:
    import pyopenjtalk

    is_available = True
except ModuleNotFoundError:
    is_available = False


class PyopenjtalkG2P(LanguageModule):
    """
    G2P Module using pyopenjtalk
    """

    def __init__(self):
        super().__init__()

    def g2p(self, text) -> List[str]:
        return ["pau"] + pyopenjtalk.g2p(text).split(" ") + ["pau"]

    def phonemes(self) -> List[str]:
        return [
            "pau",
            "I",
            "N",
            "U",
            "a",
            "b",
            "by",
            "ch",
            "cl",
            "d",
            "dy",
            "e",
            "f",
            "g",
            "gy",
            "h",
            "hy",
            "i",
            "j",
            "k",
            "ky",
            "m",
            "my",
            "n",
            "ny",
            "o",
            "p",
            "py",
            "r",
            "ry",
            "s",
            "sh",
            "t",
            "ts",
            "ty",
            "u",
            "v",
            "w",
            "y",
            "z",
            "kw",
            "gw",
        ]
