from typing import Protocol, List


class Grapheme2PhonemeModule(Protocol):
    def phonemes(self) -> List[str]:
        pass

    def g2p(self, text: str) -> List[str]:
        pass