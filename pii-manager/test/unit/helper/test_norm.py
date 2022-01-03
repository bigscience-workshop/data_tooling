import pii_manager.helper.normalizer as mod


TEST = [("the Social Security\nNumber is 34512", "the social security number is 34512")]


def test10_normalizer():
    """
    Create base object
    """
    for (text, exp) in TEST:
        assert mod.normalize(text, "en", whitespace=True, lowercase=True) == exp
