"""
Detection and validation of the identifier for Brazilian Cadastro de Pessoa
Física

It contains two check digits, so it can be validated.
"""

import re

from stdnum.br import cpf

from typing import Iterable

from pii_manager import PiiEnum


_CPF_REGEX = re.compile(r"\d{3} \. \d{3} \. \d{3} - \d{2}", flags=re.X)


def cadastro_pessoa_fisica(doc: str) -> Iterable[str]:
    """
    Brazilian número de inscrição no Cadastro de Pessoas Físicas (detect and validate)
    """
    for candidate in _CPF_REGEX.findall(doc):
        if cpf.is_valid(candidate):
            yield candidate


PII_TASKS = [(PiiEnum.GOV_ID, cadastro_pessoa_fisica)]
