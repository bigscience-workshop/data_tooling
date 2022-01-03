import pytest

from pii_manager import PiiEnum, PiiEntity
from pii_manager.helper.base import BasePiiTask
from pii_manager.helper.exception import PiiUnimplemented, InvArgException

import pii_manager.helper.base as mod


def test10_base():
    """
    Create base object
    """
    task_spec = {"pii": PiiEnum.BITCOIN_ADDRESS, "lang": "es", "name": "example"}
    task = mod.BasePiiTask(**task_spec)
    assert task.pii == PiiEnum.BITCOIN_ADDRESS
    assert task.lang == "es"
    assert task.name == "example"

    with pytest.raises(PiiUnimplemented):
        task("blah")


def test20_regex():
    """
    Test regex object
    """
    task_spec = {"pii": PiiEnum.CREDIT_CARD, "lang": "es", "name": "example"}
    task = mod.RegexPiiTask(r"\d{4}", **task_spec)

    got = list(task("number 1234 and number 3451"))
    exp = [
        PiiEntity(PiiEnum.CREDIT_CARD, 7, "1234", name="example"),
        PiiEntity(PiiEnum.CREDIT_CARD, 23, "3451", name="example"),
    ]
    assert exp == got


def test30_callable():
    """
    Test callable object
    """

    def example(i: str):
        return ["1234", "3451"]

    task_spec = {"pii": PiiEnum.CREDIT_CARD, "lang": "es", "name": "example"}
    task = mod.CallablePiiTask(example, **task_spec)

    got = list(task("number 1234 and number 3451"))
    exp = [
        PiiEntity(PiiEnum.CREDIT_CARD, 7, "1234", name="example"),
        PiiEntity(PiiEnum.CREDIT_CARD, 23, "3451", name="example"),
    ]
    assert exp == got
