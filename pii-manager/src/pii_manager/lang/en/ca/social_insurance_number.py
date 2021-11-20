"""
Detection and validation of Canadian Social Insurance Number

Since it contains a check digit, it can be validated.
"""

import re

from stdnum.ca import sin

from typing import Iterable

from pii_manager import PiiEnum


_SIN_REGEX = re.compile(r"\d{3}[-\ ]\d{3}[-\ ]\d{3}", flags=re.X)


def social_insurance_number(doc: str) -> Iterable[str]:
    """
    Canadian Social Insurance Number (detect and validate)
    """
    for candidate in _SIN_REGEX.findall(doc):
        if sin.is_valid(candidate):
            yield candidate


PII_TASKS = [(PiiEnum.GOV_ID, social_insurance_number)]
