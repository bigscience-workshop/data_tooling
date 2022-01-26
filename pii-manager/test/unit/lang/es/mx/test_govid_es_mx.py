"""
Test Mexican CURP
"""

from pii_manager import PiiEnum
from pii_manager.api import PiiManager

TEST = [
    # A valid CURP
    ("Mi número de CURP es PEPP700101HASRRD09", "Mi número de CURP es <GOV_ID>"),
    # An invalid CURP
    (
        "Mi número de CURP es PEPP700101HASRRD01",
        "Mi número de CURP es PEPP700101HASRRD01",
    ),
]


def test10_curp():
    obj = PiiManager("es", "MX", PiiEnum.GOV_ID)
    for doc, exp in TEST:
        got = obj(doc)
        assert got == exp
