"""
Detection of U.S. Social Security Number.

We just match on the number, it cannot be
validated using only the number since it does not carry a checksum
"""

from pii_manager import PiiEnum


_SSN_PATTERN = r"(?!000|666|333)0*(?:[0-6][0-9][0-9]|[0-7][0-6][0-9]|[0-7][0-7][0-2])[-\ ](?!00)[0-9]{2}[-\ ](?!0000)[0-9]{4}"


PII_TASKS = [
    (PiiEnum.GOV_ID, _SSN_PATTERN, "U.S. Social Security Number (detect only)")
]
