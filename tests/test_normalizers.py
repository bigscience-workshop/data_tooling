from ac_dc.normalizers import *

incorrect_word_substrings = ["http", "www", ".com", "href", "//"]


def test_lower_strip_sentence():
    assert lower_strip_sentence(" Hello World   ") == "hello world"


def test_remove_words_with_incorrect_substrings():
    assert (
        remove_words_with_incorrect_substrings(
            "http://www.google.com", incorrect_word_substrings
        )
        == ""
    )

    assert (
        remove_words_with_incorrect_substrings(
            "This is not an url http://google.com", incorrect_word_substrings
        )
        == "This is not an url"
    )


def test_remove_long_words():
    assert remove_long_words("This is averylongword", 5) == "This is"
