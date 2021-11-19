"""
Test Spanish Bank Accounts
"""

from pii_manager import PiiEnum
from pii_manager.api import PiiManager

TEST = [
    # A valid bank account number
    (
        "Código cuenta cliente: 2085 8720 60 1902070563",
        "Código cuenta cliente: <BANK_ACCOUNT>",
    ),
    # No spaces
    (
        "Código cuenta cliente: 20858720601902070563",
        "Código cuenta cliente: <BANK_ACCOUNT>",
    ),
    # An invalid bank account number
    (
        "Código cuenta cliente: 2085 8720 44 1902070563",
        "Código cuenta cliente: 2085 8720 44 1902070563",
    ),
]


def test10_bank_account():
    obj = PiiManager("es", "ES", PiiEnum.BANK_ACCOUNT)
    for doc, exp in TEST:
        got = obj(doc)
        assert got == exp


def test20_bank_account_undefined():
    """
    Test under another country (hence it will NOT be defined)
    """
    obj = PiiManager("es", "FR", PiiEnum.BANK_ACCOUNT)
    for doc, exp in TEST:
        got = obj(doc)
        assert got == doc
