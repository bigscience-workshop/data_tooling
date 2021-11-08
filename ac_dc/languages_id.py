import pandas as pd

# TODO: Complete this dataframe by adding for each
# language the id of the kenlm model, and the langid lib
langs_id = [
    {
        "oscar_id": "af",
        "nltk_id": "Afrikaans",
    },
    {
        "oscar_id": "als",
        "nltk_id": "Tosk Albanian",
    },
    {
        "oscar_id": "am",
        "nltk_id": "Amharic",
    },
    {
        "oscar_id": "an",
        "nltk_id": "Aragonese",
    },
    {
        "oscar_id": "ar",
        "nltk_id": "Arabic",
    },
    {
        "oscar_id": "arz",
        "nltk_id": "Egyptian Arabic",
    },
    {
        "oscar_id": "ast",
        "nltk_id": "Asturian",
    },
    {
        "oscar_id": "as",
        "nltk_id": "Assamese",
    },
    {
        "oscar_id": "av",
        "nltk_id": "Avaric",
    },
    {
        "oscar_id": "azb",
        "nltk_id": "South Azerbaijani",
    },
    {
        "oscar_id": "az",
        "nltk_id": "Azerbaijani",
    },
    {
        "oscar_id": "bar",
        "nltk_id": "Bavarian",
    },
    {
        "oscar_id": "ba",
        "nltk_id": "Bashkir",
    },
    {
        "oscar_id": "bcl",
        "nltk_id": "Central Bikol",
    },
    {
        "oscar_id": "be",
        "nltk_id": "Belarusian",
    },
    {
        "oscar_id": "bg",
        "nltk_id": "Bulgarian",
    },
    {
        "oscar_id": "bh",
        "nltk_id": "Bihari",
    },
    {
        "oscar_id": "bn",
        "nltk_id": "Bengali",
    },
    {
        "oscar_id": "bo",
        "nltk_id": "Tibetan",
    },
    {
        "oscar_id": "bpy",
        "nltk_id": "Bishnupriya",
    },
    {
        "oscar_id": "br",
        "nltk_id": "Breton",
    },
    {
        "oscar_id": "bs",
        "nltk_id": "Bosnian",
    },
    {
        "oscar_id": "bxr",
        "nltk_id": "Russia Buriat",
    },
    {
        "oscar_id": "ca",
        "nltk_id": "Catalan",
    },
    {
        "oscar_id": "cbk",
        "nltk_id": "Chavacano",
    },
    {
        "oscar_id": "ceb",
        "nltk_id": "Cebuano",
    },
    {
        "oscar_id": "ce",
        "nltk_id": "Chechen",
    },
    {
        "oscar_id": "ckb",
        "nltk_id": "Central Kurdish",
    },
    {
        "oscar_id": "cs",
        "nltk_id": "Czech",
    },
    {
        "oscar_id": "cv",
        "nltk_id": "Chuvash",
    },
    {
        "oscar_id": "cy",
        "nltk_id": "Welsh",
    },
    {
        "oscar_id": "da",
        "nltk_id": "Danish",
    },
    {
        "oscar_id": "de",
        "nltk_id": "German",
    },
    {
        "oscar_id": "diq",
        "nltk_id": "Dimli",
    },
    {
        "oscar_id": "dsb",
        "nltk_id": "Lower Sorbian",
    },
    {
        "oscar_id": "dv",
        "nltk_id": "Dhivehi",
    },
    {
        "oscar_id": "el",
        "nltk_id": "Modern Greek",
    },
    {
        "oscar_id": "eml",
        "nltk_id": "Emilian-Romagnol",
    },
    {
        "oscar_id": "en",
        "nltk_id": "English",
    },
    {
        "oscar_id": "eo",
        "nltk_id": "Esperanto",
    },
    {
        "oscar_id": "es",
        "nltk_id": "Spanish",
    },
    {
        "oscar_id": "et",
        "nltk_id": "Estonian",
    },
    {
        "oscar_id": "eu",
        "nltk_id": "Basque",
    },
    {
        "oscar_id": "fa",
        "nltk_id": "Persian",
    },
    {
        "oscar_id": "fi",
        "nltk_id": "Finnish",
    },
    {
        "oscar_id": "frr",
        "nltk_id": "Northern Frisian",
    },
    {
        "oscar_id": "fr",
        "nltk_id": "French",
    },
    {
        "oscar_id": "fy",
        "nltk_id": "Western Frisian",
    },
    {
        "oscar_id": "ga",
        "nltk_id": "Irish",
    },
    {
        "oscar_id": "gd",
        "nltk_id": "Scottish Gaelic",
    },
    {
        "oscar_id": "gl",
        "nltk_id": "Galician",
    },
    {
        "oscar_id": "gn",
        "nltk_id": "Guarani",
    },
    {
        "oscar_id": "gom",
        "nltk_id": "Goan Konkani",
    },
    {
        "oscar_id": "gu",
        "nltk_id": "Gujarati",
    },
    {
        "oscar_id": "he",
        "nltk_id": "Hebrew",
    },
    {
        "oscar_id": "hi",
        "nltk_id": "Hindi",
    },
    {
        "oscar_id": "hr",
        "nltk_id": "Croatian",
    },
    {
        "oscar_id": "hsb",
        "nltk_id": "Upper Sorbian",
    },
    {
        "oscar_id": "ht",
        "nltk_id": "Haitian",
    },
    {
        "oscar_id": "hu",
        "nltk_id": "Hungarian",
    },
    {
        "oscar_id": "hy",
        "nltk_id": "Armenian",
    },
    {
        "oscar_id": "ia",
        "nltk_id": "Interlingua",
    },
    {
        "oscar_id": "id",
        "nltk_id": "Indonesian",
    },
    {
        "oscar_id": "ie",
        "nltk_id": "Interlingue",
    },
    {
        "oscar_id": "ilo",
        "nltk_id": "Iloko",
    },
    {
        "oscar_id": "io",
        "nltk_id": "Ido",
    },
    {
        "oscar_id": "is",
        "nltk_id": "Icelandic",
    },
    {
        "oscar_id": "it",
        "nltk_id": "Italian",
    },
    {
        "oscar_id": "ja",
        "nltk_id": "Japanese",
    },
    {
        "oscar_id": "jbo",
        "nltk_id": "Lojban",
    },
    {
        "oscar_id": "jv",
        "nltk_id": "Javanese",
    },
    {
        "oscar_id": "ka",
        "nltk_id": "Georgian",
    },
    {
        "oscar_id": "kk",
        "nltk_id": "Kazakh",
    },
    {
        "oscar_id": "km",
        "nltk_id": "Central Khmer",
    },
    {
        "oscar_id": "kn",
        "nltk_id": "Kannada",
    },
    {
        "oscar_id": "ko",
        "nltk_id": "Korean",
    },
    {
        "oscar_id": "krc",
        "nltk_id": "Karachay-Balkar",
    },
    {
        "oscar_id": "ku",
        "nltk_id": "Kurdish",
    },
    {
        "oscar_id": "kv",
        "nltk_id": "Komi",
    },
    {
        "oscar_id": "kw",
        "nltk_id": "Cornish",
    },
    {
        "oscar_id": "ky",
        "nltk_id": "Kirghiz",
    },
    {
        "oscar_id": "la",
        "nltk_id": "Latin",
    },
    {
        "oscar_id": "lb",
        "nltk_id": "Luxembourgish",
    },
    {
        "oscar_id": "lez",
        "nltk_id": "Lezghian",
    },
    {
        "oscar_id": "li",
        "nltk_id": "Limburgan",
    },
    {
        "oscar_id": "lmo",
        "nltk_id": "Lombard",
    },
    {
        "oscar_id": "lo",
        "nltk_id": "Lao",
    },
    {
        "oscar_id": "lrc",
        "nltk_id": "Northern Luri",
    },
    {
        "oscar_id": "lt",
        "nltk_id": "Lithuanian",
    },
    {
        "oscar_id": "lv",
        "nltk_id": "Latvian",
    },
    {
        "oscar_id": "mai",
        "nltk_id": "Maithili",
    },
    {
        "oscar_id": "mg",
        "nltk_id": "Malagasy",
    },
    {
        "oscar_id": "mhr",
        "nltk_id": "Eastern Mari",
    },
    {
        "oscar_id": "min",
        "nltk_id": "Minangkabau",
    },
    {
        "oscar_id": "mk",
        "nltk_id": "Macedonian",
    },
    {
        "oscar_id": "ml",
        "nltk_id": "Malayalam",
    },
    {
        "oscar_id": "mn",
        "nltk_id": "Mongolian",
    },
    {
        "oscar_id": "mrj",
        "nltk_id": "Western Mari",
    },
    {
        "oscar_id": "mr",
        "nltk_id": "Marathi",
    },
    {
        "oscar_id": "ms",
        "nltk_id": "Malay",
    },
    {
        "oscar_id": "mt",
        "nltk_id": "Maltese",
    },
    {
        "oscar_id": "mwl",
        "nltk_id": "Mirandese",
    },
    {
        "oscar_id": "my",
        "nltk_id": "Burmese",
    },
    {
        "oscar_id": "myv",
        "nltk_id": "Erzya",
    },
    {
        "oscar_id": "mzn",
        "nltk_id": "Mazanderani",
    },
    {
        "oscar_id": "nah",
        "nltk_id": "Nahuatl",
    },
    {
        "oscar_id": "nap",
        "nltk_id": "Neapolitan",
    },
    {
        "oscar_id": "nds",
        "nltk_id": "Low German",
    },
    {
        "oscar_id": "ne",
        "nltk_id": "Nepali",
    },
    {
        "oscar_id": "new",
        "nltk_id": "Newari",
    },
    {
        "oscar_id": "nl",
        "nltk_id": "Dutch",
    },
    {
        "oscar_id": "nn",
        "nltk_id": "Norwegian Nynorsk",
    },
    {
        "oscar_id": "no",
        "nltk_id": "Norwegian",
    },
    {
        "oscar_id": "oc",
        "nltk_id": "Occitan",
    },
    {
        "oscar_id": "or",
        "nltk_id": "Oriya",
    },
    {
        "oscar_id": "os",
        "nltk_id": "Ossetian",
    },
    {
        "oscar_id": "pam",
        "nltk_id": "Pampanga",
    },
    {
        "oscar_id": "pa",
        "nltk_id": "Panjabi",
    },
    {
        "oscar_id": "pl",
        "nltk_id": "Polish",
    },
    {
        "oscar_id": "pms",
        "nltk_id": "Piemontese",
    },
    {
        "oscar_id": "pnb",
        "nltk_id": "Western Panjabi",
    },
    {
        "oscar_id": "ps",
        "nltk_id": "Pushto",
    },
    {
        "oscar_id": "pt",
        "nltk_id": "Portuguese",
    },
    {
        "oscar_id": "qu",
        "nltk_id": "Quechua",
    },
    {
        "oscar_id": "rm",
        "nltk_id": "Romansh",
    },
    {
        "oscar_id": "ro",
        "nltk_id": "Romanian",
    },
    {
        "oscar_id": "ru",
        "nltk_id": "Russian",
    },
    {
        "oscar_id": "sah",
        "nltk_id": "Yakut",
    },
    {
        "oscar_id": "sa",
        "nltk_id": "Sanskrit",
    },
    {
        "oscar_id": "scn",
        "nltk_id": "Sicilian",
    },
    {
        "oscar_id": "sd",
        "nltk_id": "Sindhi",
    },
    {
        "oscar_id": "sh",
        "nltk_id": "Serbo-Croatian",
    },
    {
        "oscar_id": "si",
        "nltk_id": "Sinhala",
    },
    {
        "oscar_id": "sk",
        "nltk_id": "Slovak",
    },
    {
        "oscar_id": "sl",
        "nltk_id": "Slovenian",
    },
    {
        "oscar_id": "so",
        "nltk_id": "Somali",
    },
    {
        "oscar_id": "sq",
        "nltk_id": "Albanian",
    },
    {
        "oscar_id": "sr",
        "nltk_id": "Serbian",
    },
    {
        "oscar_id": "su",
        "nltk_id": "Sundanese",
    },
    {
        "oscar_id": "sv",
        "nltk_id": "Swedish",
    },
    {
        "oscar_id": "sw",
        "nltk_id": "Swahili",
    },
    {
        "oscar_id": "ta",
        "nltk_id": "Tamil",
    },
    {
        "oscar_id": "te",
        "nltk_id": "Telugu",
    },
    {
        "oscar_id": "tg",
        "nltk_id": "Tajik",
    },
    {
        "oscar_id": "th",
        "nltk_id": "Thai",
    },
    {
        "oscar_id": "tk",
        "nltk_id": "Turkmen",
    },
    {
        "oscar_id": "tl",
        "nltk_id": "Tagalog",
    },
    {
        "oscar_id": "tr",
        "nltk_id": "Turkish",
    },
    {
        "oscar_id": "tt",
        "nltk_id": "Tatar",
    },
    {
        "oscar_id": "tyv",
        "nltk_id": "Tuvinian",
    },
    {
        "oscar_id": "ug",
        "nltk_id": "Uighur",
    },
    {
        "oscar_id": "uk",
        "nltk_id": "Ukrainian",
    },
    {
        "oscar_id": "ur",
        "nltk_id": "Urdu",
    },
    {
        "oscar_id": "uz",
        "nltk_id": "Uzbek",
    },
    {
        "oscar_id": "vec",
        "nltk_id": "Venetian",
    },
    {
        "oscar_id": "vi",
        "nltk_id": "Vietnamese",
    },
    {
        "oscar_id": "vo",
        "nltk_id": "Volapük",
    },
    {
        "oscar_id": "war",
        "nltk_id": "Waray",
    },
    {
        "oscar_id": "wa",
        "nltk_id": "Walloon",
    },
    {
        "oscar_id": "wuu",
        "nltk_id": "Wu Chinese",
    },
    {
        "oscar_id": "xal",
        "nltk_id": "Kalmyk",
    },
    {
        "oscar_id": "xmf",
        "nltk_id": "Mingrelian",
    },
    {
        "oscar_id": "yi",
        "nltk_id": "Yiddish",
    },
    {
        "oscar_id": "yo",
        "nltk_id": "Yoruba",
    },
    {
        "oscar_id": "yue",
        "nltk_id": "Yue Chinese",
    },
    {
        "oscar_id": "zh",
        "nltk_id": "Chinese",
    },
]
langs_id = pd.DataFrame(langs_id)
