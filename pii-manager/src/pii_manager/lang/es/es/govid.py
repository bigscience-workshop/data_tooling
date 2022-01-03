"""
Spanish Goverment-issued IDs:
  - DNI (Documento Nacional de Identidad)
  - NIE (Número de Identificación de Extranjero)
"""

import re

from typing import Iterable

from stdnum.es import dni, nie

from pii_manager import PiiEnum, PiiEntity
from pii_manager.helper import BasePiiTask

# regex for DNI & NIE
_DNI_PATTERN = r"\d{6,8} -? [A-KJ-NP-TV-Z]"
_NIE_PATTERN = r"[X-Z] \d{7} -? [A-KJ-NP-TV-Z]"


class SpanishDniNie(BasePiiTask):
    """
    Spanish Government-issued DNI & NIE numbers, recognize & validate
    """

    pii_name = "Spanish DNI and NIE numbers"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Compile the regexes
        self.dni = re.compile(_DNI_PATTERN, flags=re.X)
        self.nie = re.compile(_NIE_PATTERN, flags=re.X)

    def find(self, doc: str) -> Iterable[PiiEntity]:
        # DNI
        for item in self.dni.finditer(doc):
            item_value = item.group()
            if dni.is_valid(item_value):
                yield PiiEntity(
                    PiiEnum.GOV_ID,
                    item.start(),
                    item_value,
                    country=self.country,
                    name="Spanish DNI",
                )
        # NIE
        for item in self.nie.finditer(doc):
            item_value = item.group()
            if nie.is_valid(item_value):
                yield PiiEntity(
                    PiiEnum.GOV_ID,
                    item.start(),
                    item_value,
                    country=self.country,
                    name="Spanish NIE",
                )


# Task descriptor
PII_TASKS = [(PiiEnum.GOV_ID, SpanishDniNie)]
