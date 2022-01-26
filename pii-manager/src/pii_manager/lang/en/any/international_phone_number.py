"""
Detection of phone numbers written with international notation (i.e. with
prefix and country code)
"""


from pii_manager import PiiEnum

PATTERN_INT_PHONE = r"""
    (?:\+ | 00)
    (?: 9[976]\d | 8[987530]\d | 6[987]\d | 5[90]\d | 42\d |
        3[875]\d | 2[98654321]\d | 9[8543210] | 8[6421] |
        6[6543210] | 5[87654321] | 4[987654310] | 3[9643210] |
        2[70] | 7 | 1)
    [-\x20\.]?
    (?: \d{2,3} [-\x20]? ){3,4}
"""

PII_TASKS = [
    {
        "pii": PiiEnum.PHONE_NUMBER,
        "type": "regex",
        "task": PATTERN_INT_PHONE,
        "name": "international phone number",
        "doc": "detect phone numbers that use international notation. Uses context",
        "context": {"value": ["ph", "phone", "fax"], "width": [16, 0], "type": "word"},
    }
]
