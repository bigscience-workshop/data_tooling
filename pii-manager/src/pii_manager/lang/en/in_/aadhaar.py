"""
Detection and validation of Indian Aadhaar identity number

Since it contains a check digit, it can be validated.
"""

import re

from stdnum.in_ import aadhaar

from typing import Iterable

from pii_manager import PiiEnum


_AADHAAR_REGEX = re.compile(r"[2-9]\d{3}\ ?\d{4}\ ?\d{4}", flags=re.X)


def aadhaar_number(doc: str) -> Iterable[str]:
    """
    Aadhaar identity number from India (detect and validate)
    """
    for candidate in _AADHAAR_REGEX.findall(doc):
        if aadhaar.is_valid(candidate):
            yield candidate


PII_TASKS = [(PiiEnum.GOV_ID, aadhaar_number)]
