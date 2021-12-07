import re


trannum = str.maketrans("0123456789", "1111111111")

address_regex: {
    "en": {
        "en_US_street": re.compile(r"\d{1,4} [\w\s]{1,20} (?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|park|parkway|pkwy|circle|cir|boulevard|blvd)\W?(?=\s|$)"),
        "en_US_POBox": re.compile(r"P\.? ?O\.? Box \d+"),
        # AU adddresses regex: https://gist.github.com/3zzy/ea117b273c36ea62207748d6897bd552
        "en_AU_line_1": re.compile(r"\b(?:(?!\s{2,}|\$|\:|\.\d).)*\s(?:Alley|Ally|Arcade|Arc|Avenue|Ave|Boulevard|Bvd|Bypass|Bypa|Circuit|Cct|Close|Cl|Corner|Crn|Court|Ct|Crescent|Cres|Cul-de-sac|Cds|Drive|Dr|Esplanade|Esp|Green|Grn|Grove|Gr|Highway|Hwy|Junction|Jnc|Lane|Lane|Link|Link|Mews|Mews|Parade|Pde|Place|Pl|Ridge|Rdge|Road|Rd|Square|Sq|Street|St|Terrace|Tce|ALLEY|ALLY|ARCADE|ARC|AVENUE|AVE|BOULEVARD|BVD|BYPASS|BYPA|CIRCUIT|CCT|CLOSE|CL|CORNER|CRN|COURT|CT|CRESCENT|CRES|CUL-DE-SAC|CDS|DRIVE|DR|ESPLANADE|ESP|GREEN|GRN|GROVE|GR|HIGHWAY|HWY|JUNCTION|JNC|LANE|LANE|LINK|LINK|MEWS|MEWS|PARADE|PDE|PLACE|PL|RIDGE|RDGE|ROAD|RD|SQUARE|SQ|STREET|ST|TERRACE|TCE))\s.*?(?=\s{2,}"),
        "en_AU_line_2" re.compile(r"\b(?:(?!\s{2,}).)*)\b(VIC|NSW|ACT|QLD|NT|SA|TAS|WA).?\s*(\b\d{4}")
     }
}
age_regex:{
    "en": re.compile(r""\S+ years old|\S+\-years\-old|\S+ year old|\S+\-year\-old")
}
credit_card_regex = {
    "amex": re.compile(r"3[47][0-9]{13}"),
    "mastercard": re.compile(r"?:5[1-5][0-9]{2}|222[1-9]|22[3-9][0-9]|2[3-6][0-9]{2}|27[01][0-9]|2720)[0-9]{12}"),
    "visa": re.compile(r"\b([4]\d{3}[\s]\d{4}[\s]\d{4}[\s]\d{4}|[4]\d{3}[-]\d{4}[-]\d{4}[-]\d{4}|[4]\d{3}[.]\d{4}[.]\d{4}[.]\d{4}|[4]\d{3}\d{4}\d{4}\d{4})\b")
}
date_regex = {
    "default" re.compile(r"[ ][\d][\d]+[\\ /.][\d][\d][\\ /.][\d][\d]+")
}
email_regex = {
    "default": re.compile(r"[\w\.=-]+@[\w\.-]+\.[\w]{2,3}")
}
govt_id_regex = {
    "en": {
        "en_US_govt_id": re.compile(r"(?!000|666|333)0*(?:[0-6][0-9][0-9]|[0-7][0-6][0-9]|[0-7][0-7][0-2])[- ](?!00)[0-9]{2}[- ](?!0000)[0-9]{4}"),
        "en_CA_govt_id": re.compile(r"\d{3}\s\d{3}\s\d{3}"),
        "en_GB_govt_id": re.compile(r"\w{2}\s?\d{2}\s?\d{2}\s?\w|GB\s?\d{6}\s?\w|GB\d{3}\s\d{3}\s\d{2}\s\d{3}|GBGD\d{3}|GBHA\d{3}}|GB\d{3} \d{4} \d{2}(?: \d{3})?|GB(?:GD|HA)\d{3}"),
        "en_IE_govt_id": re.compile(r"IE\d[1-9]\d{5}\d[1-9]|IE\d{7}[1-9][1-9]?"),
        "en_IN_govt_id": re.compile(r"[1-9]\d{10}"),
        "en_PH_govt_id": re.compile(r"\d{2}-\d{7}-\d|\d{11}|\d{2}-\d{9}-\d|\d{4}-\d{4}-\d{4}|\d{4}-\d{7}-\d"),
    },

    "id": {
        "id_ID_govt_id": re.compile(r"\d{6}([04][1-9]|[1256][0-9]|[37][01])(0[1-9]|1[0-2])\d{6}")
    }
    "es": {
        "es_ES_govt_id": re.compile(r"(?:ES)?\d{6-8}-?[A-Z]"),
        "es_CO_govt_id": re.compile(
            r"[1-9]\d?\d{6}|8\d{8}|9\d{8}|10\d{8}|11\d{8}|12\d{8}|"),
    },
    "pt": {
        "pt_BR_govt_id": re.compile(r"\d{3}\.d{3}\.d{3}-\d{2}|\d{11}"),
        "pt_PT_govt_id": re.compile(r"PT\d{9}"),
    },
    "zh": {
        "zh_CN_govt_id": re.compile(r"\d{18}"),
        "zh_TW_govt_id": re.compile(r"[1-9]\d{9}"),
    },
    "default": {"any_govt_id": re.compile(r"\d{8}|\d{9}|\d{10}|\d{11}")
    },
}
IP_regex = {
    "default": re.compile(r"\d{1,3}[.]\d{1,3}[.]\d{1,3}[.]\d{1,3}")
}
NORP_regex = {
    "en": {re.compile(r"upper class|middle class|working class|lower class")}
}
password_regex = {
    "default": re.compile(r"[\d][\d][\d][\d][\d]+")
}
phone_regex = {
    "default": re.compile(r"[\d]?[\d]?[ -\\/.]?[ -\\/.]?[\d][\d][\d][ -\\/.]?[ -\\/.]?[\d][\d][\d][ -\\/.]?[\d][\d][\d][\d]")
}
SIN_regex: {
    "defualt": re.compile(r"[\d][\d]+[ -.][\d][\d]+[ -.][\d][\d][\d]")
}
web_address_regex = {
    "default": re.compile(r"[https:\/]*[w]?[w]?[w]?[.]?[\da-zA-Z\-]+[.][a-z]+[\/\.a-zA-Z\-\d\?=&]*")


# Code below needs to be updated/completed.

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
