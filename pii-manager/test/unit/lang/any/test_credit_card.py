"""
Test credit card numbers
"""

from pii_manager import PiiEnum
from pii_manager.api import PiiManager


TEST = [
    # A valid credit card number
    (
        "El número de la tarjeta de crédito es 4273 9666 4581 5642",
        "El número de la tarjeta de crédito es <CREDIT_CARD>",
    ),
    # Without spaces
    ("La tarjeta es 4273966645815642", "La tarjeta es <CREDIT_CARD>"),
    # With text afterwards
    (
        "El número de la tarjeta es 4273 9666 4581 5642 probablemente",
        "El número de la tarjeta es <CREDIT_CARD> probablemente",
    ),
    # With dashes
    (
        "mi tarjeta es 4273-9666-4581-5642 con caducidad 07/22",
        "mi tarjeta es <CREDIT_CARD> con caducidad 07/22",
    ),
    # Too short
    (
        "El número de la tarjeta de crédito es 4273 9666 4581",
        "El número de la tarjeta de crédito es 4273 9666 4581",
    ),
    # Not a valid credit card number
    (
        "El número de la tarjeta de crédito es 4273 9666 4581 5641",
        "El número de la tarjeta de crédito es 4273 9666 4581 5641",
    ),
]


def test10_credit_card():
    obj = PiiManager("es", None, PiiEnum.CREDIT_CARD)
    for doc, exp in TEST:
        got = obj(doc)
        assert exp == got


def test20_credit_card_stats():
    obj = PiiManager("es", None, PiiEnum.CREDIT_CARD)
    for doc, exp in TEST:
        obj(doc)
    assert obj.stats == {"calls": 6, "CREDIT_CARD": 4}
