"""
Spanish Goverment-issued IDs:
  - DNI (Documento Nacional de Identidad)
  - NIE (Número de Identificación de Extranjero)
"""

import re

from typing import Iterable

from stdnum.es import dni, nie

from pii_manager import PiiEnum


# regex for a DNI
_DNI_PATTERN = r"\d{6,8} -? [A-KJ-NP-TV-Z]"
_NIE_PATTERN = r"[X-Z] \d{7} -? [A-KJ-NP-TV-Z]"

# compiled regex
_REGEX_DNI = None
_REGEX_NIE = None


def get_govt_id(text: str) -> Iterable[str]:
    """
    Spanish DNI & NIE, recognize & validate
    """
    # Compile regex if needed
    global _REGEX_DNI, _REGEX_NIE
    if _REGEX_DNI is None:
        _REGEX_DNI = re.compile(_DNI_PATTERN, flags=re.X)
        _REGEX_NIE = re.compile(_NIE_PATTERN, flags=re.X)
    # Find all IDs
    for item in _REGEX_DNI.findall(text):
        if dni.is_valid(item):
            yield item
    for item in _REGEX_NIE.findall(text):
        if nie.is_valid(item):
            yield item


# ---------------------------------------------------------------------

PII_TASKS = [(PiiEnum.GOV_ID, get_govt_id)]
