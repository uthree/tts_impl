from typing import List, Protocol


class LanguageModule(Protocol):
    def phonemes(self) -> List[str]:
        pass

    def g2p(self, text: str) -> List[str]:
        pass
