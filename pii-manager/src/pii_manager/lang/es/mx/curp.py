"""
Detection and validation of Clave Única de Registro de Población for Mexico

It contains two check digits, so it can be validated.
"""

import re

from stdnum.mx import curp as stdnum_curp

from typing import Iterable

from pii_manager import PiiEnum


_CURP_PATTERN = r"[A-Z] [AEIOU] [A-Z]{2} \d{6} [HM] [A-Z]{5} [0-9A-Z] \d"
_CURP_REGEX = re.compile(_CURP_PATTERN, flags=re.X)


def curp(doc: str) -> Iterable[str]:
    """
    Mexican Clave Única de Registro de Población (detect and validate)
    """
    for candidate in _CURP_REGEX.findall(doc):
        if stdnum_curp.is_valid(candidate):
            yield candidate


PII_TASKS = [(PiiEnum.GOV_ID, curp)]
