"""
Test Spanish Bank Accounts
"""

from pii_manager import PiiEnum
from pii_manager.api import PiiManager

TEST = [
    # A valid DNI
    (
        "Mi DNI es 34657934-Q",
        "Mi DNI es <GOV_ID>"
    ),
    (
        "El DNI 34657934Q es válido",
        "El DNI <GOV_ID> es válido"
    ),
    # A valid NIE
    (
        "El NIE es X3465793-S",
        "El NIE es <GOV_ID>",
    ),
    # An invalid DNI
    (
        "Mi DNI es 34657934-H",
        "Mi DNI es 34657934-H"
    )
]


def test10_dni():
    obj = PiiManager("es", "ES", PiiEnum.GOV_ID)
    for doc, exp in TEST:
        got = obj(doc)
        assert got == exp
