import re


trannum = str.maketrans("0123456789", "1111111111")

# patterns from https://github.com/joke2k/faker/tree/master/faker/providers/ssn
govt_id_regex = {
    "en": {
        "en_US_govt_id": re.compile(
            "(?!000|666|333)0*(?:[0-6][0-9][0-9]|[0-7][0-6][0-9]|[0-7][0-7][0-2])[- ](?!00)[0-9]{2}[- ](?!0000)[0-9]{4}"
        ),
        "en_CA_govt_id": re.compile(r"\d{3}\s\d{3}\s\d{3}"),
        "en_GB_govt_id": re.compile(
            r"\w{2}\s?\d{2}\s?\d{2}\s?\w|GB\s?\d{6}\s?\w|GB\d{3}\s\d{3}\s\d{2}\s\d{3}|GBGD\d{3}|GBHA\d{3}}|GB\d{3} \d{4} \d{2}(?: \d{3})?|GB(?:GD|HA)\d{3}"
        ),
        "en_IE_govt_id": re.compile(r"IE\d[1-9]\d{5}\d[1-9]|IE\d{7}[1-9][1-9]?"),
        "en_IN_govt_id": re.compile(r"[1-9]\d{10}"),
        "en_PH_govt_id": re.compile(
            r"\d{2}-\d{7}-\d|\d{11}|\d{2}-\d{9}-\d|\d{4}-\d{4}-\d{4}|\d{4}-\d{7}-\d"
        ),
    },
    "zh": {
        "zh_CN_govt_id": re.compile(r"\d{18}"),
        "zh_TW_govt_id": re.compile(r"[1-9]\d{9}"),
    },
    "es": {
        "es_ES_govt_id": re.compile(r"(?:ES)?\d{6-8}-?[A-Z]"),
        "es_CO_govt_id": re.compile(
            r"[1-9]\d?\d{6}|8\d{8}|9\d{8}|10\d{8}|11\d{8}|12\d{8}|"
        ),
    },
    "pt": {
        "pt_BR_govt_id": re.compile(r"\d{3}\.d{3}\.d{3}-\d{2}|\d{11}"),
        "pt_PT_govt_id": re.compile(r"PT\d{9}"),
    },
    "default": {"any_govt_id": re.compile(r"\d{8}|\d{9}|\d{10}|\d{11}")},
}


def apply_regex_govt_id_anonymization(sentence: str, lang_id: str) -> str:
    regex_lang_id = lang_id if lang_id in govt_id_regex else "default"
    for regex in list(govt_id_regex[regex_lang_id].values()):
        matched = False
        for ent in regex.findall(sentence):
            if not isinstance(ent, str):
                continue
            ent2 = ent.translate(trannum)
            sentence = sentence.replace(ent, ent2)
            matched = True
        if matched:
            break
    return sentence
