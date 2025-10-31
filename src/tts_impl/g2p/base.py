from typing import Protocol

import torch


class LanguageModule(Protocol):
    def phonemes(self) -> list[str]:
        pass

    def g2p(self, text: str) -> list[str]:
        pass


def remove_duplicates(lst):
    return list(sorted(list(set(lst))))


def pad(ids: list[int], length: int = 100, pad_id: int = 0) -> list[int]:
    while len(ids) < length:
        ids.append(pad_id)
    ids = ids[:length]
    return ids


class Grapheme2Phoneme:
    def __init__(self, languages: dict[str, LanguageModule]):
        self.language_modules = languages
        self.languages = list(self.language_modules.keys())

        # extract all phonemes
        self.phonemes = []
        for m in self.language_modules.values():
            self.phonemes += m.phonemes()
        self.phonemes = remove_duplicates(self.phonemes)
        self.phonemes = ["<PAD>"] + self.phonemes

    def _lang_id_single(self, language: str) -> int:
        return self.languages.index(language)

    def _g2p_single(self, text: str, language: str) -> list[str]:
        return self.language_modules[language].g2p(text)

    def _p2id_single(self, phoneme_seq: list[str]) -> list[int]:
        r = []
        for p in phoneme_seq:
            r.append(self.phonemes.index(p))
        return r

    def encode(
        self, transcriptions: list[str], languages: list[str], length: int = 100
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        encode transcriptions, language identifiers to `torch.Tensor`.

        Returns:
            token_ids: dtype=long, shape=(batch_size, length)
            lengths: dtyoe=long, shape=(batch_size)
            language_ids: dtype=long, shape=(batch_size)
        """

        token_ids = []
        tokens_lengths = []
        language_ids = []
        for t, l in zip(transcriptions, languages):
            phonemes = self._p2id_single(self._g2p_single(t, l))
            tokens_length = min(len(phonemes), length)
            token_ids.append(pad(phonemes, length))
            language_ids.append(self._lang_id_single(l))
            tokens_lengths.append(tokens_length)
        token_ids = torch.LongTensor(token_ids)
        language_ids = torch.LongTensor(language_ids)
        tokens_lengths = torch.LongTensor(tokens_lengths)
        return token_ids, tokens_lengths, language_ids
