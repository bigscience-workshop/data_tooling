import pytest
from ac_dc.filters import *


strip_characters = (
    "' 0123456789¯_%$§½¼¾×|†~\"—±′–'°−{}[]·'?,./<>!@#^&*()+-‑=:;，：,...�`→¶.…‘’”"
)

special_characters = (
    "' 0123456789¯_%$§½¼¾×|†~\"—±′–'°−{}[]·'?,./<>!@#^&*()+-‑=:;，：,...�`→¶.…‘’”",
)


def test_get_words_from_sentence():
    sentence = "This is a test sentence."
    expected = ["this", "is", "a", "test", "sentence"]
    print(">>>>>>>>>>>>", get_words_from_sentence(sentence, strip_characters))
    assert get_words_from_sentence(sentence, strip_characters) == expected


def test_check_empty():
    assert check_empty("", strip_characters) is False


def test_check_special_characters():
    special_characters_cutoff = 0.4

    assert (
        check_special_characters(
            """In Which We Are Introduced to Winnie the Pooh and Some Bees and the Stories Begin
Winnie-the-Pooh is out of honey, so he and Christopher Robin attempt to trick some bees out of theirs, with disastrous results.""",
            special_characters,
            special_characters_cutoff,
        )
        is True
    )

    assert (
        check_special_characters(
            "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\",
            special_characters,
            special_characters_cutoff,
        )
        is False
    )

    assert (
        check_special_characters(
            "12314 12312 1231231", special_characters, special_characters_cutoff
        )
        is False
    )



def test_check_stopwords():
    stopwords = [
        "a",
        "the",
        "and",
        "of",
        "to",
        "in",
        "is",
        "it",
        "for",
        "on",
        "as",
        "an",
    ]
    stopwords_cutoff = 0.4

    assert (
        check_stopwords(
            """In Which We Are Introduced to Winnie the Pooh and Some Bees and the Stories Begin
Winnie-the-Pooh is out of honey, so he and Christopher Robin attempt to trick some bees out of theirs, with disastrous results.""",
            strip_characters,
            stopwords,
            stopwords_cutoff,
        )
        is True
    )

    assert (
        check_stopwords(
            "12314 12312 1231231", strip_characters, stopwords, stopwords_cutoff
        )
        is True
    )

    assert (
        check_stopwords(
            "In a the they of to in is it for on as an",
            strip_characters,
            stopwords,
            stopwords_cutoff,
        )
        is False
    )
