import pytest
import torch
from tts_impl.g2p.base import Grapheme2Phoneme, pad, remove_duplicates


def test_remove_duplicates():
    """Test remove_duplicates function."""
    input_list = ["a", "b", "c", "a", "d", "b"]
    result = remove_duplicates(input_list)

    # Should remove duplicates and sort
    assert result == ["a", "b", "c", "d"]
    assert len(result) == 4


def test_pad_shorter():
    """Test pad function when list is shorter than target length."""
    input_list = [1, 2, 3]
    result = pad(input_list, length=10, pad_id=0)

    assert len(result) == 10
    assert result[:3] == [1, 2, 3]
    assert result[3:] == [0] * 7


def test_pad_longer():
    """Test pad function when list is longer than target length."""
    input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = pad(input_list, length=5, pad_id=0)

    assert len(result) == 5
    assert result == [1, 2, 3, 4, 5]


def test_pad_exact():
    """Test pad function when list length equals target length."""
    input_list = [1, 2, 3, 4, 5]
    result = pad(input_list, length=5, pad_id=0)

    assert len(result) == 5
    assert result == [1, 2, 3, 4, 5]


def test_pad_custom_pad_id():
    """Test pad function with custom pad_id."""
    input_list = [1, 2, 3]
    result = pad(input_list, length=6, pad_id=-1)

    assert result == [1, 2, 3, -1, -1, -1]


# Mock LanguageModule for testing
class MockLanguageModule:
    def phonemes(self):
        return ["a", "b", "c", "d"]

    def g2p(self, text):
        # Simple mock: split text into characters
        return list(text)


def test_g2p_initialization():
    """Test Grapheme2Phoneme initialization."""
    languages = {
        "lang1": MockLanguageModule(),
        "lang2": MockLanguageModule(),
    }
    g2p = Grapheme2Phoneme(languages)

    assert "lang1" in g2p.languages
    assert "lang2" in g2p.languages
    assert "<PAD>" in g2p.phonemes
    # Check that phonemes from both modules are included
    assert "a" in g2p.phonemes
    assert "b" in g2p.phonemes


def test_g2p_lang_id_single():
    """Test language ID retrieval."""
    languages = {
        "lang1": MockLanguageModule(),
        "lang2": MockLanguageModule(),
    }
    g2p = Grapheme2Phoneme(languages)

    lang1_id = g2p._lang_id_single("lang1")
    lang2_id = g2p._lang_id_single("lang2")

    assert lang1_id != lang2_id
    assert lang1_id in [0, 1]
    assert lang2_id in [0, 1]


def test_g2p_single():
    """Test single text to phoneme conversion."""
    languages = {"test": MockLanguageModule()}
    g2p = Grapheme2Phoneme(languages)

    phonemes = g2p._g2p_single("abc", "test")

    assert phonemes == ["a", "b", "c"]


def test_p2id_single():
    """Test phoneme to ID conversion."""
    languages = {"test": MockLanguageModule()}
    g2p = Grapheme2Phoneme(languages)

    phoneme_seq = ["a", "b", "c"]
    ids = g2p._p2id_single(phoneme_seq)

    # IDs should be valid indices
    assert all(isinstance(id, int) for id in ids)
    assert all(0 <= id < len(g2p.phonemes) for id in ids)
    # Same phoneme should have same ID
    assert ids[0] == g2p._p2id_single(["a"])[0]


def test_encode_single():
    """Test encoding single transcription."""
    languages = {"test": MockLanguageModule()}
    g2p = Grapheme2Phoneme(languages)

    token_ids, tokens_lengths, language_ids = g2p.encode(
        transcriptions=["abc"], languages=["test"], length=10
    )

    assert token_ids.shape == (1, 10)
    assert tokens_lengths.shape == (1,)
    assert language_ids.shape == (1,)
    assert tokens_lengths[0] == 3  # "abc" has 3 characters


def test_encode_batch():
    """Test encoding batch of transcriptions."""
    languages = {"test": MockLanguageModule()}
    g2p = Grapheme2Phoneme(languages)

    token_ids, tokens_lengths, language_ids = g2p.encode(
        transcriptions=["ab", "abcd"], languages=["test", "test"], length=10
    )

    assert token_ids.shape == (2, 10)
    assert tokens_lengths.shape == (2,)
    assert language_ids.shape == (2,)
    assert tokens_lengths[0] == 2  # "ab" has 2 characters
    assert tokens_lengths[1] == 4  # "abcd" has 4 characters


def test_encode_multiple_languages():
    """Test encoding with multiple languages."""

    class MockLangModule1:
        def phonemes(self):
            return ["x", "y", "z"]

        def g2p(self, text):
            return ["x"] * len(text)

    class MockLangModule2:
        def phonemes(self):
            return ["p", "q", "r"]

        def g2p(self, text):
            return ["p"] * len(text)

    languages = {
        "lang1": MockLangModule1(),
        "lang2": MockLangModule2(),
    }
    g2p = Grapheme2Phoneme(languages)

    token_ids, tokens_lengths, language_ids = g2p.encode(
        transcriptions=["abc", "def"], languages=["lang1", "lang2"], length=10
    )

    assert token_ids.shape == (2, 10)
    assert (
        language_ids[0] != language_ids[1]
    )  # Different languages should have different IDs


def test_encode_truncation():
    """Test that encoding truncates long sequences."""
    languages = {"test": MockLanguageModule()}
    g2p = Grapheme2Phoneme(languages)

    long_text = "a" * 100
    token_ids, tokens_lengths, language_ids = g2p.encode(
        transcriptions=[long_text], languages=["test"], length=10
    )

    assert token_ids.shape == (1, 10)
    assert tokens_lengths[0] == 10  # Should be truncated to length
