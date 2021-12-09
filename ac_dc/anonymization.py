import re, regex


trannum = str.maketrans("0123456789", "1111111111")

address_regex = {
    "en": {
        "en_US": [
            (
                re.compile(
                    r"P\.? ?O\.? Box \d+|\d{1,4} [\w\s]{1,20} (?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|park|parkway|pkwy|circle|cir|boulevard|blvd)\W?(?=\s|$)"
                ),
                None,
            )
        ],
        # getting an unbalanced paranthesis in en_AU
        #        "en_AU": [(re.compile(r"\b(?:(?!\s{2,}).)*)\b(VIC|NSW|ACT|QLD|NT|SA|TAS|WA).?\s*(\b\d{4}|\b(?:(?!\s{2,}|\$|\:|\.\d).)*\s(?:Alley|Ally|Arcade|Arc|Avenue|Ave|Boulevard|Bvd|Bypass|Bypa|Circuit|Cct|Close|Cl|Corner|Crn|Court|Ct|Crescent|Cres|Cul-de-sac|Cds|Drive|Dr|Esplanade|Esp|Green|Grn|Grove|Gr|Highway|Hwy|Junction|Jnc|Lane|Lane|Link|Link|Mews|Mews|Parade|Pde|Place|Pl|Ridge|Rdge|Road|Rd|Square|Sq|Street|St|Terrace|Tce|ALLEY|ALLY|ARCADE|ARC|AVENUE|AVE|BOULEVARD|BVD|BYPASS|BYPA|CIRCUIT|CCT|CLOSE|CL|CORNER|CRN|COURT|CT|CRESCENT|CRES|CUL-DE-SAC|CDS|DRIVE|DR|ESPLANADE|ESP|GREEN|GRN|GROVE|GR|HIGHWAY|HWY|JUNCTION|JNC|LANE|LANE|LINK|LINK|MEWS|MEWS|PARADE|PDE|PLACE|PL|RIDGE|RDGE|ROAD|RD|SQUARE|SQ|STREET|ST|TERRACE|TCE))\s.*?(?=\s{2,}"), None,)],
    },
    "zh": [
        (
            regex.compile(
                r"((\p{Han}{1,3}(自治区|省))?\p{Han}{1,4}((?<!集)市|县|州)\p{Han}{1,10}[路|街|道|巷](\d{1,3}[弄|街|巷])?\d{1,4}号)"
            ),
            None,
        ),
        (
            regex.compile(
                r"(?<zipcode>(^\d{5}|^\d{3})?)(?<city>\D+[縣市])(?<district>\D+?(市區|鎮區|鎮市|[鄉鎮市區]))(?<others>.+)"
            ),
            None,
        ),
    ],
}
age_regex = {
    "en": [
        (
            re.compile(
                r"\S+ years old|\S+\-years\-old|\S+ year old|\S+\-year\-old|born [ ][\d][\d]+[\\ /.][\d][\d][\\ /.][\d][\d]+|died [ ][\d][\d]+[\\ /.][\d][\d][\\ /.][\d][\d]+"
            ),
            None,
        )
    ],
    "zh": [(regex.compile(r"\d{1,3}歲|岁"), None)],
}

# IBAN - see https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/iban_patterns.py which is under MIT

# IBAN parts format
CC = "[A-Z]{2}"  # country code
CK = "[0-9]{2}[ ]?"  # checksum
BOS = "^"
EOS = "$"  # end of string

A = "[A-Z][ ]?"
A2 = "([A-Z][ ]?){2}"
A3 = "([A-Z][ ]?){3}"
A4 = "([A-Z][ ]?){4}"

C = "[a-zA-Z0-9][ ]?"
C2 = "([a-zA-Z0-9][ ]?){2}"
C3 = "([a-zA-Z0-9][ ]?){3}"
C4 = "([a-zA-Z0-9][ ]?){4}"

N = "[0-9][ ]?"
N2 = "([0-9][ ]?){2}"
N3 = "([0-9][ ]?){3}"
N4 = "([0-9][ ]?){4}"

# WIP - fix the country codes and group by languages
# move this to financial record
iban_regex = {
    # Albania (8n, 16c) ALkk bbbs sssx cccc cccc cccc cccc
    "al_AL": "(AL)" + CK + N4 + N4 + C4 + C4 + C4 + C4,
    # Andorra (8n, 12c) ADkk bbbb ssss cccc cccc cccc
    "ad_AD": "(AD)" + CK + N4 + N4 + C4 + C4 + C4,
    # Austria (16n) ATkk bbbb bccc cccc cccc
    "en_AT": "(AT)" + CK + N4 + N4 + N4 + N4,
    # Azerbaijan    (4c,20n) AZkk bbbb cccc cccc cccc cccc cccc
    "az_AZ": "(AZ)" + CK + C4 + N4 + N4 + N4 + N4 + N4,
    # Bahrain   (4a,14c)    BHkk bbbb cccc cccc cccc cc
    "ar_BH": "(BH)" + CK + A4 + C4 + C4 + C4 + C2,
    # Belarus (4c, 4n, 16c)   BYkk bbbb aaaa cccc cccc cccc cccc
    "bel_BY": "(BY)" + CK + C4 + N4 + C4 + C4 + C4 + C4,
    # Belgium (12n)   BEkk bbbc cccc ccxx
    "fr_BE": "(BE)" + CK + N4 + N4 + N4,
    # Bosnia and Herzegovina    (16n)   BAkk bbbs sscc cccc ccxx
    "bos_BA": "(BA)" + CK + N4 + N4 + N4 + N4,
    # Brazil (23n,1a,1c) BRkk bbbb bbbb ssss sccc cccc ccct n
    "pt_BR": "(BR)" + CK + N4 + N4 + N4 + N4 + N4 + N3 + A + C,
    # Bulgaria  (4a,6n,8c)  BGkk bbbb ssss ttcc cccc cc
    "bg_BG": "(BG)" + CK + A4 + N4 + N + N + C2 + C4 + C2,
    # Costa Rica    (18n)   CRkk 0bbb cccc cccc cccc cc (0 = always zero)
    "es_CR": "(CR)" + CK + "[0]" + N3 + N4 + N4 + N4 + N2,
    # Croatia   (17n)   HRkk bbbb bbbc cccc cccc c
    "hr_HR": "(HR)" + CK + N4 + N4 + N4 + N4 + N,
    # Cyprus    (8n,16c)    CYkk bbbs ssss cccc cccc cccc cccc
    "el_CY": "(CY)" + CK + N4 + N4 + C4 + C4 + C4 + C4,
    # Czech Republic    (20n)   CZkk bbbb ssss sscc cccc cccc
    "cz_CZ": "(CZ)" + CK + N4 + N4 + N4 + N4 + N4,
    # Denmark   (14n)   DKkk bbbb cccc cccc cc
    "dan_DK": "(DK)" + CK + N4 + N4 + N4 + N2,
    # Dominican Republic    (4a,20n)    DOkk bbbb cccc cccc cccc cccc cccc
    "es_DO": "(DO)" + CK + A4 + N4 + N4 + N4 + N4 + N4,
    # EAt Timor    (19n) TLkk bbbc cccc cccc cccc cxx
    "tl_TL": "(TL)" + CK + N4 + N4 + N4 + N4 + N3,
    # Estonia   (16n) EEkk bbss cccc cccc cccx
    "ee_EE": "(EE)" + CK + N4 + N4 + N4 + N4,
    # Faroe Islands    (14n) FOkk bbbb cccc cccc cx
    "FO": "(FO)" + CK + N4 + N4 + N4 + N2,
    # Finland   (14n) FIkk bbbb bbcc cccc cx
    "fi_FI": "(FI)" + CK + N4 + N4 + N4 + N2,
    # France    (10n,11c,2n) FRkk bbbb bsss sscc cccc cccc cxx
    "fr_FR": "(FR)" + CK + N4 + N4 + N2 + C2 + C4 + C4 + C + N2,
    # Georgia   (2c,16n)  GEkk bbcc cccc cccc cccc cc
    "ge_GE": "(GE)" + CK + C2 + N2 + N4 + N4 + N4 + N2,
    # Germany   (18n) DEkk bbbb bbbb cccc cccc cc
    "de_DE": "(DE)" + CK + N4 + N4 + N4 + N4 + N2,
    # Gibraltar (4a,15c)  GIkk bbbb cccc cccc cccc ccc
    "GI": "(GI)" + CK + A4 + C4 + C4 + C4 + C3,
    # Greece    (7n,16c)  GRkk bbbs sssc cccc cccc cccc ccc
    "el_GR": "(GR)" + CK + N4 + N3 + C + C4 + C4 + C4 + C3,
    # Greenland     (14n) GLkk bbbb cccc cccc cc
    "kl_GL": "(GL)" + CK + N4 + N4 + N4 + N2,
    # Guatemala (4c,20c)  GTkk bbbb mmtt cccc cccc cccc cccc
    "es_GT": "(GT)" + CK + C4 + C4 + C4 + C4 + C4 + C4,
    # Hungary   (24n) HUkk bbbs sssx cccc cccc cccc cccx
    "hu_HU": "(HU)" + CK + N4 + N4 + N4 + N4 + N4 + N4,
    # Iceland   (22n) ISkk bbbb sscc cccc iiii iiii ii
    "is_IS": "(IS)" + CK + N4 + N4 + N4 + N4 + N4 + N2,
    # Ireland   (4c,14n)  IEkk aaaa bbbb bbcc cccc cc
    "en_IE": "(IE)" + CK + C4 + N4 + N4 + N4 + N2,
    # Israel (19n) ILkk bbbn nncc cccc cccc ccc
    "hb_IL": "(IL)" + CK + N4 + N4 + N4 + N4 + N3,
    # Italy (1a,10n,12c)  ITkk xbbb bbss sssc cccc cccc ccc
    "it_IT": "(IT)" + CK + A + N3 + N4 + N3 + C + C3 + C + C4 + C3,
    # Jordan    (4a,22n)  JOkk bbbb ssss cccc cccc cccc cccc cc
    "ar_JO": "(JO)" + CK + A4 + N4 + N4 + N4 + N4 + N4 + N2,
    # Kazakhstan    (3n,13c)  KZkk bbbc cccc cccc cccc
    "kz_KZ": "(KZ)" + CK + N3 + C + C4 + C4 + C4,
    # Kosovo    (4n,10n,2n)   XKkk bbbb cccc cccc cccc
    "xk_XK": "(XK)" + CK + N4 + N4 + N4 + N4,
    # Kuwait    (4a,22c)  KWkk bbbb cccc cccc cccc cccc cccc cc
    "ar_KW": "(KW)" + CK + A4 + C4 + C4 + C4 + C4 + C4 + C2,
    # Latvia    (4a,13c)  LVkk bbbb cccc cccc cccc c
    "lv_LV": "(LV)" + CK + A4 + C4 + C4 + C4 + C,
    # Lebanon   (4n,20c)  LBkk bbbb cccc cccc cccc cccc cccc
    "lb_LB": "(LB)" + CK + N4 + C4 + C4 + C4 + C4 + C4,
    # de_LiechteNtein (5n,12c)  LIkk bbbb bccc cccc cccc c
    "li_LI": "(LI)" + CK + N4 + N + C3 + C4 + C4 + C,
    # Lithuania (16n) LTkk bbbb bccc cccc cccc
    "lt_LT": "(LT)" + CK + N4 + N4 + N4 + N4,
    # Luxembourg    (3n,13c)  LUkk bbbc cccc cccc cccc
    "lu_LU": "(LU)" + CK + N3 + C + C4 + C4 + C4,
    # Malta (4a,5n,18c)   MTkk bbbb ssss sccc cccc cccc cccc ccc
    "mt_MT": "(MT)" + CK + A4 + N4 + N + C3 + C4 + C4 + C4 + C3,
    # Mauritania    (23n) MRkk bbbb bsss sscc cccc cccc cxx
    "mr_MR": "(MR)" + CK + N4 + N4 + N4 + N4 + N4 + N3,
    # Mauritius (4a,19n,3a)   MUkk bbbb bbss cccc cccc cccc 000m mm
    "mu_MU": "(MU)" + CK + A4 + N4 + N4 + N4 + N4 + N3 + A,
    # Moldova   (2c,18c)  MDkk bbcc cccc cccc cccc cccc
    "md_MD": "(MD)" + CK + C4 + C4 + C4 + C4 + C4,
    # Monaco    (10n,11c,2n)  MCkk bbbb bsss sscc cccc cccc cxx
    "mc_MC": "(MC)" + CK + N4 + N4 + N2 + C2 + C4 + C4 + C + N2,
    # Montenegro    (18n) MEkk bbbc cccc cccc cccc xx
    "me_ME": "(ME)" + CK + N4 + N4 + N4 + N4 + N2,
    # Netherlands   (4a,10n)  NLkk bbbb cccc cccc cc
    "nl_NL": "(NL)" + CK + A4 + N4 + N4 + N2,
    # North Macedonia   (3n,10c,2n)   MKkk bbbc cccc cccc cxx
    "mk_MK": "(MK)" + CK + N3 + C + C4 + C4 + C + N2,
    # Norway    (11n) NOkk bbbb cccc ccx
    "no_NO": "(NO)" + CK + N4 + N4 + N3,
    # Pakistan  (4c,16n)  PKkk bbbb cccc cccc cccc cccc
    "pk_PK": "(PK)" + CK + C4 + N4 + N4 + N4 + N4,
    # Palestinian territories   (4c,21n)  PSkk bbbb xxxx xxxx xccc cccc cccc c
    "ps_PS": "(PS)" + CK + C4 + N4 + N4 + N4 + N4 + N,
    # Poland    (24n) PLkk bbbs sssx cccc cccc cccc cccc
    "pl_PL": "(PL)" + CK + N4 + N4 + N4 + N4 + N4 + N4,
    # Portugal  (21n) PTkk bbbb ssss cccc cccc cccx x
    "pt_PT": "(PT)" + CK + N4 + N4 + N4 + N4 + N,
    # Qatar (4a,21c)  QAkk bbbb cccc cccc cccc cccc cccc c
    "ar_QA": "(QA)" + CK + A4 + C4 + C4 + C4 + C4 + C,
    # Romania   (4a,16c)  ROkk bbbb cccc cccc cccc cccc
    "ro_RO": "(RO)" + CK + A4 + C4 + C4 + C4 + C4,
    # San Marino    (1a,10n,12c)  SMkk xbbb bbss sssc cccc cccc ccc
    "sm": "(SM)" + CK + A + N3 + N4 + N3 + C + C4 + C4 + C3,
    # Saudi Arabia  (2n,18c)  SAkk bbcc cccc cccc cccc cccc
    "ar_SA": "(SA)" + CK + N2 + C2 + C4 + C4 + C4 + C4,
    # Serbia    (18n) RSkk bbbc cccc cccc cccc xx
    "rs_RS": "(RS)" + CK + N4 + N4 + N4 + N4 + N2,
    # Slovakia  (20n) SKkk bbbb ssss sscc cccc cccc
    "sk_SK": "(SK)" + CK + N4 + N4 + N4 + N4 + N4,
    # Slovenia  (15n) SIkk bbss sccc cccc cxx
    "si_SI": "(SI)" + CK + N4 + N4 + N4 + N3,
    # Spain (20n) ESkk bbbb ssss xxcc cccc cccc
    "es_ES": "(ES)" + CK + N4 + N4 + N4 + N4 + N4,
    # Sweden    (20n) SEkk bbbc cccc cccc cccc cccc
    "se_SE": "(SE)" + CK + N4 + N4 + N4 + N4 + N4,
    # Switzerland   (5n,12c)  CHkk bbbb bccc cccc cccc c
    "gsw_CH": "(CH)" + CK + N4 + N + C3 + C4 + C4 + C,
    # Tunisia   (20n) TNkk bbss sccc cccc cccc cccc
    "ar_TN": "(TN)" + CK + N4 + N4 + N4 + N4 + N4,
    # Turkey    (5n,17c)  TRkk bbbb bxcc cccc cccc cccc cc
    "tr_TR": "(TR)" + CK + N4 + N + C3 + C4 + C4 + C4 + C2,
    # United Arab Emirates  (3n,16n)  AEkk bbbc cccc cccc cccc ccc
    "ar_AE": "(AE)" + CK + N4 + N4 + N4 + N4 + N3,
    # United Kingdom (4a,14n) GBkk bbbb ssss sscc cccc cc
    "en_GB": "(GB)" + CK + A4 + N4 + N4 + N4 + N2,
    # Vatican City  (3n,15n)  VAkk bbbc cccc cccc cccc cc
    "it_VA": "(VA)" + CK + N4 + N4 + N4 + N4 + N2,
    # Virgin Islands, British   (4c,16n)  VGkk bbbb cccc cccc cccc cccc
    "en_VG": "(VG)" + CK + C4 + N4 + N4 + N4 + N4,
}


# ABA routing from https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/aba_routing_recognizer.py which is licensed under the MIT License
financial_record_regex = {
    "en": [
        (
            re.compile(r"\b[0123678]\d{3}-\d{4}-\d\b"),
            (
                "aba",
                "routing",
                "abarouting",
                "association",
                "bankrouting",
            ),
        )
    ],
    # for credit card, getting a "nothing to repeat at position 15"
    # (re.compile(r"3[47][0-9]{13}|?:5[1-5][0-9]{2}|222[1-9]|22[3-9][0-9]|2[3-6][0-9]{2}|27[01][0-9]|2720)[0-9]{12}|\b([4]\d{3}[\s]\d{4}[\s]\d{4}[\s]\d{4}|[4]\d{3}[-]\d{4}[-]\d{4}[-]\d{4}|[4]\d{3}[.]\d{4}[.]\d{4}[.]\d{4}|[4]\d{3}\d{4}\d{4}\d{4})\b"), ("credit card", "american express", "visa", "mastercard")),
}

email_regex = {"default": [(re.compile(r"[\w\.=-]+@[\w\.-]+\.[\w]{2,3}"), None)]}

# see https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/au_abn_recognizer.py which is licensed under MIT
# see also https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/us_passport_recognizer.py
# see also https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/medical_license_recognizer.py
# see also https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/es_nif_recognizer.py

govt_id_regex = {
    "en": {
        "en_US": [
            (
                re.compile(
                    r"(?!000|666|333)0*(?:[0-6][0-9][0-9]|[0-7][0-6][0-9]|[0-7][0-7][0-2])[- ](?!00)[0-9]{2}[- ](?!0000)[0-9]{4}"
                ),
                None,
            ),
            (
                re.compile(r"(\b[0-9]{9}\b)"),
                (
                    "us",
                    "united",
                    "states",
                    "passport",
                    "passport#",
                    "travel",
                    "document",
                ),
            ),
            (
                re.compile(r"[a-zA-Z]{2}\d{7}|[a-zA-Z]{1}9\d{7}"),
                ("medical", "certificate", "DEA"),
            ),
        ],
        "en_CA": [(re.compile(r"\d{3}\s\d{3}\s\d{3}"), None)],
        "en_GB": [
            (
                re.compile(
                    r"\w{2}\s?\d{2}\s?\d{2}\s?\w|GB\s?\d{6}\s?\w|GB\d{3}\s\d{3}\s\d{2}\s\d{3}|GBGD\d{3}|GBHA\d{3}}|GB\d{3} \d{4} \d{2}(?: \d{3})?|GB(?:GD|HA)\d{3}"
                ),
                None,
            )
        ],
        "en_IE": [(re.compile(r"IE\d[1-9]\d{5}\d[1-9]|IE\d{7}[1-9][1-9]?"), None)],
        "en_IN": [(re.compile(r"[1-9]\d{10}"), None)],
        "en_PH": [
            (
                re.compile(
                    r"\d{2}-\d{7}-\d|\d{11}|\d{2}-\d{9}-\d|\d{4}-\d{4}-\d{4}|\d{4}-\d{7}-\d"
                ),
                None,
            )
        ],
        "en_AU": [
            (
                re.compile(r"\b\d{2}\s\d{3}\s\d{3}\s\d{3}\b|\b\d{11}\b"),
                ("australian business number", "abn"),
            )
        ],
    },
    "id": {
        "id_ID": [
            (
                re.compile(
                    r"\d{6}([04][1-9]|[1256][0-9]|[37][01])(0[1-9]|1[0-2])\d{6}"
                ),
                None,
            )
        ]
    },
    "es": {
        "es_ES": [
            (re.compile(r"(?:ES)?\d{6-8}-?[A-Z]"), None),
            (
                re.compile(r"\b[0-9]?[0-9]{7}[-]?[A-Z]\b"),
                ("documento nacional de identidad", "DNI", "NIF", "identificación"),
            ),
        ],
        "es_CO": [
            (re.compile(r"[1-9]\d?\d{6}|8\d{8}|9\d{8}|10\d{8}|11\d{8}|12\d{8}|"), None)
        ],
    },
    "pt": {
        "pt_BR": [(re.compile(r"\d{3}\.d{3}\.d{3}-\d{2}|\d{11}"), None)],
        "pt_PT": [(re.compile(r"PT\d{9}"), None)],
    },
    "zh": [
        (
            regex.compile(
                r"(?:[16][1-5]|2[1-3]|3[1-7]|4[1-6]|5[0-4])\d{4}(?:19|20)\d{2}(?:(?:0[469]|11)(?:0[1-9]|[12][0-9]|30)|(?:0[13578]|1[02])(?:0[1-9]|[12][0-9]|3[01])|02(?:0[1-9]|[12][0-9]))\d{3}[\dXx]"
            ),
            None,
        ),
        (
            regex.compile(
                r"(^[EeKkGgDdSsPpHh]\d{8}$)|(^(([Ee][a-fA-F])|([DdSsPp][Ee])|([Kk][Jj])|([Mm][Aa])|(1[45]))\d{7}$)"
            ),
            None,
        ),
        (
            regex.compile(
                r"((\d{4}(| )\d{4}(| )\d{4}$)|([a-zA-Z][1-2]{1}[0-9]{8})|([0-3]{1}\d{8}))"
            ),
            None,
        ),
    ],
    "default": [(re.compile(r"\d{8}|\d{9}|\d{10}|\d{11}"), None)],
}

# should we move license plate to govt_id?
# ("LICENSE_PLATE", regex.compile('^(?:[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领 A-Z]{1}[A-HJ-NP-Z]{1}(?:(?:[0-9]{5}[DF])|(?:[DF](?:[A-HJ-NP-Z0-9])[0-9]{4})))|(?:[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领 A-Z]{1}[A-Z]{1}[A-HJ-NP-Z0-9]{4}[A-HJ-NP-Z0-9 挂学警港澳]{1})$'), None, None, None),
# ("LICENSE_PLATE", regex.compile('\b[A-Z]{3}-\d{4}\b'), None, None, None),


# too broad - might need a context word
IP_regex = {
    "default": [
        (re.compile(r"\d{1,3}[.]\d{1,3}[.]\d{1,3}[.]\d{1,3}"), ("address", "ip"))
    ],
}
NORP_regex = {
    "en": [(re.compile(r"upper class|middle class|working class|lower class"), None)],
}

# I think this is too broad. I put a context word around it.
password_regex = {
    "en": [(re.compile(r"password: [\d][\d][\d][\d][\d]+"), ("password",))],
}

# We will want a context word around it to make it more specific. might be too broad.
phone_regex = {
    "en": [
        (
            re.compile(
                r"[\d]?[\d]?[ -\\/.]?[ -\\/.]?[\d][\d][\d][ -\\/.]?[ -\\/.]?[\d][\d][\d][ -\\/.]?[\d][\d][\d][\d]"
            ),
            ("ph", "phone", "fax"),
        )
    ],
    "zh": [
        (
            regex.compile(
                r"(0?\d{2,4}-[1-9]\d{6,7})|({\+86|086}-| ?1[3-9]\d{9} , ([\+0]?86)?[\-\s]?1[3-9]\d{9})"
            ),
            None,
        ),
        (
            regex.compile(
                r"((\d{4}(| )\d{4}(| )\d{4}$)|([a-zA-Z][1-2]{1}[0-9]{8})|([0-3]{1}\d{8}))((02|03|037|04|049|05|06|07|08|089|082|0826|0836|886 2|886 3|886 37|886 4|886 49|886 5|886 6|886 7|886 8|886 89|886 82|886 826|886 836|886 9|886-2|886-3|886-37|886-4|886-49|886-5|886-6|886-7|886-8|886-89|886-82|886-826|886-836)(| |-)\d{4}(| |-)\d{4}$)|((09|886 9|886-9)(| |-)\d{2}(|-)\d{2}(|-)\d{1}(|-)\d{3})"
            ),
            None,
        ),
    ],
}

# Does this correspond to CA id?
SIN_regex = {"defualt": re.compile(r"[\d][\d]+[ -.][\d][\d]+[ -.][\d][\d][\d]")}

# will this only match https, what about http?
domain_name_regex = {
    "default": [
        (
            re.compile(
                r"[https:\/]*[w]?[w]?[w]?[.]?[\da-zA-Z\-]+[.][a-z]+[\/\.a-zA-Z\-\d\?=&]*"
            ),
            None,
        )
    ],
}


tag_2_regex = [
    ("DOMAIN_NAME", (domain_name_regex, False)),
    ("PHONE", (phone_regex, True)),
    ("PASSWORD", (password_regex, True)),
    ("NORP", (NORP_regex, False)),
    ("AGE", (age_regex, False)),
    ("ADDRESS", (address_regex, True)),
    ("EMAIL", (email_regex, True)),
    ("IP_ADDRESS", (IP_regex, True)),
    ("GOVT_ID", (govt_id_regex, True)),
    ("FIN_ID", (financial_record_regex, True)),
]


# Code below needs to be updated/completed.


def apply_regex_anonymization(
    sentence: str, lang_id: str, context_window: int = 20
) -> str:
    ner = {}
    lang_id = regex_lang_id.split("_")[0]
    if lang_id in ("zh", "ko", "ja"):
        sentence_set = set(sentence.lower())
    else:
        sentence_set = set(sentence.lower().split(" "))
    for tag, regex_group_and_anonymize_condition in tag_2_regex:
        regex_group, anonymize_condition = regex_group_and_anonymize_condition
        for regex_dict in regex_group.get(lang_id, regex_group.get("default", [])):
            if isinstance(regex_dict, dict):
                regex_list = regex_dict.get(regex_lang_id, [])
            else:
                regex_list = regex_dict
            match = False
            for regex, context in regex_list:
                found_context = False
                if context:
                    for c in context:
                        if c in sentence_set:
                            found_context = True
                            break
                    if not found_context:
                        continue
                for ent in regex.findall(sentence):
                    if not isinstance(ent, str):
                        continue
                    if found_context:
                        i = sentence.index(ent)
                        j = i + len(ent)
                        len_sentence = len(sentence)
                        left = sentence[max(0, i - context_window) : i].lower()
                        right = sentence[
                            j : min(len_sentence, j + context_window)
                        ].lower()
                        found_context = False
                        for c in context:
                            if c in left or c in right:
                                found_context = True
                                break
                        if not found_context:
                            continue
                    if anonymize_condition:
                        sentence = sentence.replace(ent, f" <{tag}> ")
                        ner[f"<{tag}>"] = tag
                    else:
                        ner[ent.strip()] = tag
                    match = True
                if match:
                    break
    return sentence, ner
