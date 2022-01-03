"""
Test Portuguese NIF & CC
"""

from pii_manager import PiiEnum, PiiEntity
from pii_manager.api import PiiManager

TEST = [
    # A valid NIF
    (
        "Meu NIF é PT 123 456 789",
        "Meu NIF é <GOV_ID>",
        [PiiEntity(PiiEnum.GOV_ID, 10, "PT 123 456 789", "pt", "Portuguese NIF")],
    ),
    # A NIF without spacing or prefix
    (
        "O NIF 123456789 é valido",
        "O NIF <GOV_ID> é valido",
        [PiiEntity(PiiEnum.GOV_ID, 6, "123456789", "pt", "Portuguese NIF")],
    ),
    # A valid CC
    (
        "O CC é 00000000 0 ZZ4",
        "O CC é <GOV_ID>",
        [PiiEntity(PiiEnum.GOV_ID, 7, "00000000 0 ZZ4", "pt", "Portuguese CC")],
    ),
    # An invalid NIF
    ("Meu NIF é PT 123 456 788", "Meu NIF é PT 123 456 788", []),
]


def test10_nif_cc():
    obj = PiiManager("pt", "PT", PiiEnum.GOV_ID)
    for doc, exp, _ in TEST:
        got = obj(doc)
        assert got == exp


def test20_nif_cc_extract():
    obj = PiiManager("pt", "PT", PiiEnum.GOV_ID, mode="extract")
    for doc, _, exp in TEST:
        got = list(obj(doc))
        assert got == exp
