
from .piienum import PiiEnum


class PiiEntity:
    '''
    A detected PII entity
    '''

    __slots__ = 'pos', 'elem', 'value'

    def __init__(self, elem: PiiEnum, pos: int, value: str):
        self.elem = elem
        self.pos = pos
        self.value = value

    def __len__(self):
        return len(self.value)

    def __repr__(self):
        return f'<PiiEntity {self.elem.name}:{self.pos}:{self.value}>'
