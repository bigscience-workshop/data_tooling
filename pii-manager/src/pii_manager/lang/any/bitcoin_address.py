"""
Find valid bitcoin addresses
1. Obtain candidates, by using a generic regex expression
2. Validate candidates by
    - using a more exact regex
    - validating the number through the Luhn algorithm
"""

import re

from typing import Iterable

from stdnum import bitcoin

from pii_manager import PiiEnum

# ----------------------------------------------------------------------------

# regex for the three types of bitcoin addresses
_BITCOIN_PATTERN = (
    r"( [13] ["
    + bitcoin._base58_alphabet
    + "]{25,34}"
    + "| bc1 ["
    + bitcoin._bech32_alphabet
    + "]{8,87})"
)

_REGEX_BITCOIN = re.compile(_BITCOIN_PATTERN, flags=re.X)


def bitcoin_address(text: str) -> Iterable[str]:
    """
    Bitcoin addresses (P2PKH, P2SH and Bech32), recognize & validate
    """
    # Find and validate candidates
    for ba in _REGEX_BITCOIN.findall(text):
        if bitcoin.is_valid(ba):
            yield ba


# ---------------------------------------------------------------------

PII_TASKS = [(PiiEnum.BITCOIN_ADDRESS, bitcoin_address)]
