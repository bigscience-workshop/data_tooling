from .piienum import PiiEnum

from typing import Dict


class PiiEntity:
    """
    A detected PII entity. It contains as fields:
      * elem, a PiiEnum that describes the type of the detected PII
      * pos, the character position of the PII inside the passed document
      * value, the string containing the PII
      * country, the country this PII is applicable to (or None)
      * name, an optional specific name for this PII
    """

    __slots__ = "elem", "pos", "value", "country", "name"

    def __init__(
        self, elem: PiiEnum, pos: int, value: str, country: str = None, name: str = None
    ):
        self.elem = elem
        self.pos = pos
        self.value = value
        self.country = country
        self.name = name

    def __len__(self):
        return len(self.value)

    def __repr__(self):
        return f"<PiiEntity {self.elem.name}:{self.pos}:{self.value}:{self.country}:{self.name}>"

    def __eq__(self, other):
        return (
            self.elem == other.elem
            and self.pos == other.pos
            and self.value == other.value
            and self.country == other.country
            and self.name == other.name
        )

    def to_json(self) -> Dict:
        """
        Return the object data as a dict that can then be serialised as JSON
        """
        return piientity_asdict(self)


def piientity_asdict(pii: PiiEntity, name: bool = None, country: bool = None) -> Dict:
    """
    Create a dictionary from a PiiEntity object
     :param country: add country information: always (True), never (False),
        only if defined (None)
    """
    n = {"name": pii.name} if name or country is None and pii.name else {}
    d = {"type": pii.elem.name, **n, "value": pii.value, "pos": pii.pos}
    if country or country is None and pii.country:
        d["country"] = pii.country
    return d
