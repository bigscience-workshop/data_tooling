"""
Test Spanish DNI & NIE
"""

from pii_manager import PiiEnum, PiiEntity
from pii_manager.api import PiiManager

TEST = [
    # A valid DNI
    (
        "Mi DNI es 34657934-Q",
        "Mi DNI es <GOV_ID>",
        [PiiEntity(PiiEnum.GOV_ID, 10, "34657934-Q", "es", "Spanish DNI")],
    ),
    # A DNI without dash
    (
        "El DNI 34657934Q es válido",
        "El DNI <GOV_ID> es válido",
        [PiiEntity(PiiEnum.GOV_ID, 7, "34657934Q", "es", "Spanish DNI")],
    ),
    # A valid NIE
    (
        "El NIE es X3465793-S",
        "El NIE es <GOV_ID>",
        [PiiEntity(PiiEnum.GOV_ID, 10, "X3465793-S", "es", "Spanish NIE")],
    ),
    # An invalid DNI
    ("Mi DNI es 34657934-H", "Mi DNI es 34657934-H", []),
]


def test10_dni():
    obj = PiiManager("es", "ES", PiiEnum.GOV_ID)
    for doc, exp, _ in TEST:
        got = obj(doc)
        assert got == exp


def test20_dni_extract():
    obj = PiiManager("es", "ES", PiiEnum.GOV_ID, mode="extract")
    for doc, _, exp in TEST:
        got = list(obj(doc))
        assert got == exp
