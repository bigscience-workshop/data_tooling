"""
Test Brazilian CPF
"""

from pii_manager import PiiEnum
from pii_manager.api import PiiManager

TEST = [
    # A valid CPF
    ("O número do CPF é 263.946.533-30", "O número do CPF é <GOV_ID>"),
    # An invalid CPF
    ("O número do CPF é 000.000.000-12", "O número do CPF é 000.000.000-12"),
]


def test10_cpf():
    obj = PiiManager("pt", "BR", PiiEnum.GOV_ID)
    for doc, exp in TEST:
        got = obj(doc)
        assert got == exp
