

from io import StringIO

from pii_manager import PiiEnum
from pii_manager.api import PiiManager


TEST = ('El número de la tarjeta de crédito es 4273 9666 4581 5642',
        'El número de la tarjeta de crédito es <CREDIT_CARD>',)

def test10_constructor():
    obj = PiiManager('es', None, PiiEnum.CREDIT_CARD)
    assert obj.tasks[0].pii == PiiEnum.CREDIT_CARD


def test20_info():
    obj = PiiManager('es', None, PiiEnum.CREDIT_CARD)
    info = obj.task_info()

    exp = {PiiEnum.CREDIT_CARD: "Credit card numbers for most international credit cards (recognize & validate)"}
    assert info == exp


def test20_call():
    obj = PiiManager('es', None, PiiEnum.CREDIT_CARD)
    anon = obj(TEST[0])
    assert anon == TEST[1]
