import pytest

from pii_manager import PiiEnum
from pii_manager.lang import LANG_ANY
from pii_manager.helper.base import BasePiiTask
import pii_manager.helper.taskdict as mod


def test10_lang():
    """
    Check the presence of a minimum set of languages in the dict
    """
    taskdict = mod.get_taskdict()
    for lang in (LANG_ANY, "en", "es", "fr", "zh"):
        assert lang in taskdict


def test20_lang_any():
    """
    Test taskdict contents
    """
    taskdict = mod.get_taskdict()
    assert len(taskdict) >= 2

    # Find one element that should be there
    tasklist = taskdict[LANG_ANY][PiiEnum.CREDIT_CARD.name]

    # Check its contents
    assert isinstance(tasklist, list)
    assert isinstance(tasklist[0], dict)

    elem = tasklist[0]
    assert len(elem) == 7
    assert sorted(elem.keys()) == [
        "country",
        "doc",
        "lang",
        "name",
        "pii",
        "task",
        "type",
    ]

    assert elem["country"] is None
    assert elem["lang"] == "any"
    assert elem["pii"] == PiiEnum.CREDIT_CARD
    assert issubclass(elem["task"], BasePiiTask)
    assert elem["type"] == "PiiTask"
    assert elem["name"] == "credit card"
    assert (
        elem["doc"]
        == "Credit card numbers for most international credit cards (detect & validate)"
    )


_TASK = [
    (PiiEnum.CREDIT_CARD, r"\d16", "a toy Credit Card example"),
    {
        "pii": PiiEnum.CREDIT_CARD,
        "type": "regex",
        "task": r"\d16",
        "doc": "a toy Credit Card example",
    },
]


@pytest.mark.parametrize("task", _TASK)
def test31_task(task):
    """
    Check the function parsing a PII_TASKS list, single entry
    """
    subdict = mod.build_subdict([task], lang="en")

    assert len(subdict) == 1
    assert isinstance(subdict, dict)

    assert PiiEnum.CREDIT_CARD.name in subdict
    tasklist = subdict[PiiEnum.CREDIT_CARD.name]
    assert isinstance(tasklist, list)
    assert len(tasklist) == 1

    assert tasklist[0] == {
        "pii": PiiEnum.CREDIT_CARD,
        "lang": "en",
        "country": None,
        "type": "regex",
        "task": r"\d16",
        "name": "regex for credit_card",
        "doc": "a toy Credit Card example",
    }


def test32_task_multiple_same():
    """
    Check the function parsing a PII_TASKS list, multiple entries, same task
    """
    subdict = mod.build_subdict(_TASK, lang="es")

    assert len(subdict) == 1
    assert isinstance(subdict, dict)

    assert PiiEnum.CREDIT_CARD.name in subdict
    tasklist = subdict[PiiEnum.CREDIT_CARD.name]
    assert isinstance(tasklist, list)
    assert len(tasklist) == 2

    exp = {
        "pii": PiiEnum.CREDIT_CARD,
        "lang": "es",
        "country": None,
        "type": "regex",
        "task": r"\d16",
        "name": "regex for credit_card",
        "doc": "a toy Credit Card example",
    }
    assert tasklist[0] == exp
    assert tasklist[1] == exp


def test33_task_multiple_different():
    """
    Check the function parsing a PII_TASKS list, multiple different entries
    """

    def toy_example(x):
        """another toy example"""
        return x

    PII_TASKS = [
        (PiiEnum.CREDIT_CARD, r"\d16", "a toy Credit Card example"),
        (PiiEnum.BITCOIN_ADDRESS, toy_example),
    ]
    subdict = mod.build_subdict(PII_TASKS, "zh")

    assert isinstance(subdict, dict)
    assert len(subdict) == 2

    exp1 = {
        "pii": PiiEnum.CREDIT_CARD,
        "lang": "zh",
        "country": None,
        "type": "regex",
        "task": r"\d16",
        "name": "regex for credit_card",
        "doc": "a toy Credit Card example",
    }
    assert subdict[PiiEnum.CREDIT_CARD.name] == [exp1]

    exp2 = {
        "pii": PiiEnum.BITCOIN_ADDRESS,
        "lang": "zh",
        "country": None,
        "type": "callable",
        "task": toy_example,
        "name": "toy example",
        "doc": "another toy example",
    }

    assert subdict[PiiEnum.BITCOIN_ADDRESS.name] == [exp2]


def test34_subdict_lang_country():
    """
    Check the function parsing a PII_TASKS list, w/ language & country
    """

    def callable_example(x):
        return x

    PII_TASKS = [
        (PiiEnum.CREDIT_CARD, r"\d16", "a toy Credit Card example"),
        (PiiEnum.BITCOIN_ADDRESS, callable_example),
    ]
    subdict = mod.build_subdict(PII_TASKS, "en", "in")

    assert len(subdict) == 2

    exp1 = {
        "pii": PiiEnum.CREDIT_CARD,
        "lang": "en",
        "country": "in",
        "type": "regex",
        "task": r"\d16",
        "name": "regex for credit_card",
        "doc": "a toy Credit Card example",
    }
    assert subdict[PiiEnum.CREDIT_CARD.name] == [exp1]

    exp2 = {
        "pii": PiiEnum.BITCOIN_ADDRESS,
        "lang": "en",
        "country": "in",
        "type": "callable",
        "task": callable_example,
        "name": "callable example",
    }

    assert subdict[PiiEnum.BITCOIN_ADDRESS.name] == [exp2]


def test40_subdict_simplified_err():
    """
    Check the function parsing a PII_TASKS list of simplified tasks with errors
    """
    # Not a tuple
    PII_TASKS = [r"\d16"]
    with pytest.raises(mod.InvPiiTask):
        mod.build_subdict(PII_TASKS, "fr")

    # A tuple plus not a tuple
    PII_TASKS = [(PiiEnum.CREDIT_CARD, r"\d{16}", "a toy Credit Card example"), r"\d16"]
    with pytest.raises(mod.InvPiiTask):
        mod.build_subdict(PII_TASKS, "zh")

    # A tuple without a valid PiiEnum
    PII_TASKS = [("not a PiiEnum", r"\d{16}", "a toy Credit Card example")]
    with pytest.raises(mod.InvPiiTask):
        mod.build_subdict(PII_TASKS, "es")


def test41_subdict_full_err():
    """
    Check the function parsing a PII_TASKS list of full tasks with errors
    """
    # Empty dict
    PII_TASKS = [{}]
    with pytest.raises(mod.InvPiiTask):
        mod.build_subdict(PII_TASKS, "fr")

    # invalid pii field
    PII_TASKS = [{"pii": "not a valid PiiEnum", "type": "regex", "task": r"\d{16}"}]
    with pytest.raises(mod.InvPiiTask):
        mod.build_subdict(PII_TASKS, "fr")

    # invalid type field
    PII_TASKS = [
        {"pii": PiiEnum.CREDIT_CARD, "type": "not a valid type", "task": r"\d{16}"}
    ]
    with pytest.raises(mod.InvPiiTask):
        mod.build_subdict(PII_TASKS, "fr")

    # No task
    PII_TASKS = [{"pii": PiiEnum.CREDIT_CARD, "type": "regex"}]
    with pytest.raises(mod.InvPiiTask):
        mod.build_subdict(PII_TASKS, "fr")

    # Invalid task descriptor for a regex
    PII_TASKS = [{"pii": PiiEnum.CREDIT_CARD, "type": "regex", "task": lambda x: x}]
    with pytest.raises(mod.InvPiiTask):
        mod.build_subdict(PII_TASKS, "fr")

    # Invalid task descriptor for a callable
    PII_TASKS = [{"pii": PiiEnum.CREDIT_CARD, "type": "callable", "task": r"\d{16}"}]
    with pytest.raises(mod.InvPiiTask):
        mod.build_subdict(PII_TASKS, "fr")

    # Invalid task descriptor for a class
    PII_TASKS = [{"pii": PiiEnum.CREDIT_CARD, "type": "PiiTask", "task": lambda x: x}]
    with pytest.raises(mod.InvPiiTask):
        mod.build_subdict(PII_TASKS, "fr")

    # No language
    PII_TASKS = [{"pii": PiiEnum.CREDIT_CARD, "type": "regex", "task": r"\d{16}"}]
    with pytest.raises(mod.InvPiiTask):
        mod.build_subdict(PII_TASKS, lang=None)
