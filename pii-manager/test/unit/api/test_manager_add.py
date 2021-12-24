from io import StringIO

from pii_manager import PiiEnum
from pii_manager.api import PiiManager
from pii_manager.lang import LANG_ANY

# regex for credit card type
# https://www.regular-expressions.info/creditcard.html
_TOY_PATTERN = r"""ABCD [\d-]5"""



# ---------------------------------------------------------------------

DUMMY_DISEASE = {
    "pii": PiiEnum.DISEASE,
    "type": "regex",
    "task": r"""\b(cancer|leuka?emia|aids)\b""",
    "lang": LANG_ANY,
    "name": "disease names",
    "doc": "a toy example to match some disease names"
}



TEST = (
    "Alvin had cancer detected and his email is alvin@anywhere.com",
    "Alvin had <DISEASE> detected and his email is <EMAIL_ADDRESS>"
)


def test10_call():
    obj = PiiManager("en", None, PiiEnum.EMAIL_ADDRESS)
    obj.add_tasks([DUMMY_DISEASE])

    anon = obj(TEST[0])
    assert anon == TEST[1]
