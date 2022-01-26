"""
Detection and validation of Australian Tax File Number (TFN).

"""
import re

from stdnum.au import tfn

from typing import Iterable

from pii_manager import PiiEnum


_TFN_PATTERN = r"\b (?: \d{3} \s \d{3} \s \d{3} | \d{8,9} ) \b"
_TFN_REGEX = re.compile(_TFN_PATTERN, flags=re.X)


def tax_file_number(doc: str) -> Iterable[str]:
    """
    Australian Tax File Number (detect and validate)
    """
    for candidate in _TFN_REGEX.findall(doc):
        if tfn.is_valid(candidate):
            yield candidate


PII_TASKS = [(PiiEnum.GOV_ID, tax_file_number)]
