"""
Detection of various Chinese PII elements
"""


from pii_manager import PiiEnum


_PATTERNS = {
    "STREET_ADDRESS": r"""(\p{Han}{1,4} (自治区|省))?
        \p{Han}{1,4}
        ((?<!集)市|县|州)
        \p{Han}{1,10}
        [路|街|道|巷]
        (\d{1,3}[弄|街|巷])?
        \d{1,4}号""",
    "PHONE_NUMBER": r"""(?<!\d) (?:
            0? \d{2,4} - [1-9] \d{6,7}
            |
            (?: [\+0]? 86 )? [\-\s]? 1[3-9] \d{9}
         ) (?!\d)""",  # Home Phone, Cell Phone
    #'LICENSE_PLATE': ['''([粤沪京湘京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新][\w]{6,7})'''],
    "DISEASE": r"(?: 癌症 | 心臟疾病 | 阿爾茨海默氏病 | 老年癡呆症 )",
}


PII_TASKS = [
    (PiiEnum.STREET_ADDRESS, _PATTERNS["STREET_ADDRESS"], "Chinese street addresses"),
    (
        PiiEnum.PHONE_NUMBER,
        _PATTERNS["PHONE_NUMBER"],
        "Chinese Home Phone & Cell Phone numbers",
    ),
    (PiiEnum.DISEASE, _PATTERNS["DISEASE"], "Disease names for Chinese"),
]
