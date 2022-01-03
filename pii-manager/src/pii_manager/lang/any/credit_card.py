"""
Find valid credit card numbers:
1. Obtain candidates, by using a generic regex expression
2. Validate candidates by
    - using a more exact regex
    - validating the number through the Luhn algorithm
"""

import re

from stdnum import luhn

from typing import Iterable

from pii_manager import PiiEnum, PiiEntity
from pii_manager.helper import BasePiiTask


# ----------------------------------------------------------------------------

# base regex to detect candidates to credit card numbers
_CREDIT_PATTERN_BASE = r"\b \d (?:\d[ -]?){14} \d \b"

# full regex for credit card type
# https://www.regular-expressions.info/creditcard.html
_CREDIT_PATTERN = r"""4[0-9]{12}(?:[0-9]{3})? |
                      (?:5[1-5][0-9]{2}|222[1-9]|22[3-9][0-9]|2[3-6][0-9]{2}|27[01][0-9]|2720)[0-9]{12} |
                      3[47][0-9]{13} |
                      3(?:0[0-5]|[68][0-9])[0-9]{11} |
                      6(?:011|5[0-9]{2})[0-9]{12} |
                      (?:2131|1800|35\d{3})\d{11}"""

# compiled regexes
_REGEX_CC_BASE = None
_REGEX_CC_FULL = None


class CreditCard(BasePiiTask):
    """
    Credit card numbers for most international credit cards (detect & validate)
    """

    pii_name = "credit card"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Compile the credit card regexes
        global _REGEX_CC_FULL, _REGEX_CC_BASE
        if _REGEX_CC_FULL is None:
            _REGEX_CC_BASE = re.compile(_CREDIT_PATTERN_BASE, flags=re.VERBOSE)
            _REGEX_CC_FULL = re.compile(_CREDIT_PATTERN, flags=re.VERBOSE)

    def find(self, doc: str) -> Iterable[PiiEntity]:
        # First find candidates
        for cc in _REGEX_CC_BASE.finditer(doc):
            cc_value = cc.group()
            # strip spaces and dashes
            strip_cc = re.sub(r"[ -]+", "", cc_value)
            # now validate the credit card number
            if re.fullmatch(_REGEX_CC_FULL, strip_cc) and luhn.is_valid(strip_cc):
                yield PiiEntity(
                    PiiEnum.CREDIT_CARD, cc.start(), cc_value, name=CreditCard.pii_name
                )


# ---------------------------------------------------------------------

PII_TASKS = [(PiiEnum.CREDIT_CARD, CreditCard)]
