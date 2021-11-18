import pytest
import anonymization


@pytest.mark.parametrize("sentence,lang_id,expected", [
    ("hello world", "en", "hello world"),
    ("my government id is 123 123 123", "en", "my government id is 111 111 111"),
])
def test_govt_anonymization(sentence, lang_id, expected):
    replaced_text = anonymization.apply_regex_govt_id_anonymization(sentence, lang_id)
    assert replaced_text == expected