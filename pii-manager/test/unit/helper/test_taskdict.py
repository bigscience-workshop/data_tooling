

from pii_manager import PiiEnum
from pii_manager.helper import TASK_ANY

import pii_manager.helper as mod


def test_lang_all():
    taskdict = mod.get_taskdict()
    assert len(taskdict) >= 2
    elem = taskdict[TASK_ANY][PiiEnum.CREDIT_CARD.name]
    assert len(elem) == 2
    assert elem[0] == PiiEnum.CREDIT_CARD


def test_lang_es():
    taskdict = mod.get_taskdict()
    assert 'es' in taskdict
