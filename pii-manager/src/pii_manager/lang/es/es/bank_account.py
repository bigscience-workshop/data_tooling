"""
Spanish bank account numbers (CCC - código cuenta cliente)

Note: **NOT** IBAN numbers, those are country (& language) independent
"""

import re

from typing import Iterable

from stdnum.es import ccc

from pii_manager import PiiEnum

# ----------------------------------------------------------------------------

# regex for a Código Cuenta Cliente, with optional spaces separating the pieces
_CCC_PATTERN = r"\d{4}\s?\d{4}\s?\d{2}\s?\d{10}"

# compiled regex
_REGEX_CCC = None


def spanish_bank_ccc(text: str) -> Iterable[str]:
    """
    Spanish Bank Accounts (código cuenta cliente, 10-digit code, pre-IBAN), recognize & validate
    """
    # Compile regex if needed
    global _REGEX_CCC
    if _REGEX_CCC is None:
        _REGEX_CCC = re.compile(_CCC_PATTERN, flags=re.X)
    # Find all CCCs
    for item in _REGEX_CCC.findall(text):
        if ccc.is_valid(item):
            yield item


# ---------------------------------------------------------------------

PII_TASKS = [(PiiEnum.BANK_ACCOUNT, spanish_bank_ccc)]
