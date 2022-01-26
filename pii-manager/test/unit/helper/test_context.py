"""
Test the context checking function
"""
import pytest

import pii_manager.helper.context as mod
from pii_manager.helper.exception import InvArgException

TEST_TRUE = [
    ("a special number is 34512", ["special number"]),
    ("a special number is 34512", "special number"),
    ("a special number is 34512", {"value": "special number"}),
    ("a special number is 34512", {"value": "special number", "width": 20}),
    ("a special number is 34512", {"value": "special number", "width": [20, 20]}),
    (
        "a special number is 34512",
        {"value": "special number", "width": [20, 20], "type": "string"},
    ),
    (
        "special numbering is 34512",
        {"value": "special number", "width": 20, "type": "string"},
    ),
    (
        "a special number is 34512",
        {"value": "special number", "width": [20, 20], "type": "word"},
    ),
    (
        "a special number is 34512",
        {"value": r"special\snumber", "width": [20, 20], "type": "regex"},
    ),
    (
        "a special number is 34512",
        {"value": r"(?:special|standard)\snumber", "width": [20, 20], "type": "regex"},
    ),
    (
        "special numbering is 34512",
        {"value": r"\bspecial\snumber(?:ing)?\b", "width": 30, "type": "regex"},
    ),
]


TEST_FALSE = [
    # non-matching string
    ("a special tiny number is 34512", ["special number"]),
    # too small context width
    ("a special number is 34512", {"value": "special number", "width": 8}),
    # not full words
    (
        "special numbering 34512",
        {"value": "special number", "width": 20, "type": "word"},
    ),
    # not a valid extended regex (it has whitespace)
    (
        "special numbering 34512",
        {"value": "special number", "width": 20, "type": "regex"},
    ),
    # marked as a string, while it should be regex
    (
        "special numbering is 34512",
        {"value": r"\bspecial\snumber(?:ing)?\b", "width": 30, "type": "string"},
    ),
]


TEST_ERROR = [
    None,
    "",
    ["special number", ""],
    {"value": "special number", "width": 20, "type": "not-a-type"},
]


def test10_context_true():
    """
    Check valid contexts
    """
    for (text, context) in TEST_TRUE:
        spec = mod.context_spec(context)
        assert mod.context_check(text, spec, 20) is True


def test20_context_false():
    """
    Check invalid contexts
    """
    for (text, context) in TEST_FALSE:
        spec = mod.context_spec(context)
        assert mod.context_check(text, spec, 20) is False


def test20_context_error():
    """
    Check invalid context spec
    """
    for context in TEST_ERROR:
        with pytest.raises(InvArgException):
            mod.context_spec(context)
