"""
Detection and validation of Australian business number (ABN).

"""
import re

from stdnum.au import abn

from typing import Iterable

from pii_manager import PiiEnum


_ABN_PATTERN = r"\b (?: \d{2} \s \d{3} \s \d{3} \s \d{3} | \d{11} ) \b"
_ABN_REGEX = re.compile(_ABN_PATTERN, flags=re.X)


def australian_business_number(doc: str) -> Iterable[str]:
    """
    Australian Business Number (detect and validate)
    """
    for candidate in _ABN_REGEX.findall(doc):
        if abn.is_valid(candidate):
            yield candidate


PII_TASKS = [(PiiEnum.GOV_ID, australian_business_number)]
