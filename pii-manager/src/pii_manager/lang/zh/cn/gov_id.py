"""
Detection of various government-issued IDs for China:
 - Resident Identification Card number (this can be validated)
 - Passport number (this cannot)
"""

import re
from typing import Iterable

from pii_manager import PiiEnum

from stdnum.cn import ric


# Detect candidates (separately) for RIC and passport-like numbers
_GOV_ID_PATTERN = r"(?<!\d) (?: (\d{18}) | ( (?:G|D|S|P|H|M) \d{8} ) ) (?!\d)"


_GOV_ID_REGEX = re.compile(_GOV_ID_PATTERN, flags=re.X)


def ric_or_passport(doc: str) -> Iterable[str]:
    """
    Chinese government-issued identifiers:
      - RIC (Resident Identification Card number), detect and validate
      - Passport number, detect only
    """
    for g in _GOV_ID_REGEX.finditer(doc):
        if g.group(1) and ric.is_valid(g.group(1)):
            yield g.group(1)
        elif g.group(2):
            yield g.group(2)


PII_TASKS = [(PiiEnum.GOV_ID, ric_or_passport)]
