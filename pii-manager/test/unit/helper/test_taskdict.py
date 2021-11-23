import pytest

from pii_manager import PiiEnum
from pii_manager.helper import TASK_ANY
from pii_manager.helper.base import BasePiiTask
from pii_manager.helper.exception import InvArgException
import pii_manager.helper.taskdict as mod


def test10_lang_all():
    """
    Test taskdict contents
    """
    taskdict = mod.get_taskdict()
    assert len(taskdict) >= 2
    # Find one element that should be there
    elem = taskdict[TASK_ANY][PiiEnum.CREDIT_CARD.name]
    # Check its contents
    assert len(elem) == 4
    assert elem[0] == TASK_ANY
    assert elem[1] is None
    assert elem[2] == PiiEnum.CREDIT_CARD
    assert issubclass(elem[3], BasePiiTask)


def test20_lang():
    """
    Check the presence of languages in the dict
    """
    taskdict = mod.get_taskdict()
    assert TASK_ANY in taskdict
    assert "en" in taskdict
    assert "es" in taskdict
    assert "fr" in taskdict


def test31_subdict():
    """
    Check the function parsing a PII_TASKS list, single entry
    """
    PII_TASKS = [(PiiEnum.CREDIT_CARD, r"\d16", "a toy Credit Card esample")]
    subdict = mod.build_subdict(PII_TASKS)
    assert len(subdict) == 1
    exp = (None, None, PiiEnum.CREDIT_CARD, r"\d16", "a toy Credit Card esample")
    assert subdict[PiiEnum.CREDIT_CARD.name] == exp


def test32_subdict():
    """
    Check the function parsing a PII_TASKS list, multiple entries
    """
    PII_TASKS = [
        (PiiEnum.CREDIT_CARD, r"\d16", "a toy Credit Card esample"),
        (PiiEnum.BITCOIN_ADDRESS, lambda x: x),
    ]
    subdict = mod.build_subdict(PII_TASKS)
    assert len(subdict) == 2
    exp1 = (None, None, PiiEnum.CREDIT_CARD, r"\d16", "a toy Credit Card esample")
    exp2 = (None, None, PiiEnum.BITCOIN_ADDRESS, PII_TASKS[1][1])
    assert subdict[PiiEnum.CREDIT_CARD.name] == exp1
    assert subdict[PiiEnum.BITCOIN_ADDRESS.name] == exp2


def test33_subdict():
    """
    Check the function parsing a PII_TASKS list, w/ language & country
    """
    PII_TASKS = [
        (PiiEnum.CREDIT_CARD, r"\d16", "a toy Credit Card esample"),
        (PiiEnum.BITCOIN_ADDRESS, lambda x: x),
    ]
    subdict = mod.build_subdict(PII_TASKS, "en", "in")
    assert len(subdict) == 2
    exp1 = ("en", "in", PiiEnum.CREDIT_CARD, r"\d16", "a toy Credit Card esample")
    exp2 = ("en", "in", PiiEnum.BITCOIN_ADDRESS, PII_TASKS[1][1])
    assert subdict[PiiEnum.CREDIT_CARD.name] == exp1
    assert subdict[PiiEnum.BITCOIN_ADDRESS.name] == exp2


def test34_subdict_err():
    """
    Check the function parsing a PII_TASKS list, bad list
    """
    PII_TASKS = [r"\d16"]
    with pytest.raises(InvArgException):
        mod.build_subdict(PII_TASKS)

    PII_TASKS = [(PiiEnum.CREDIT_CARD, r"\d16", "a toy Credit Card esample"), r"\d16"]
    with pytest.raises(InvArgException):
        mod.build_subdict(PII_TASKS)

    PII_TASKS = [("not a PiiEnum", r"\d16", "a toy Credit Card esample")]
    with pytest.raises(InvArgException):
        mod.build_subdict(PII_TASKS)
