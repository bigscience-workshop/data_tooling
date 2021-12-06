import pandas as pd


langs_id = [
    {
        "lang": "Afrikaans",
        "oscar_id": "af",
        "stopwords_id": "af",
        "badwords_id": None,
        "fasttext_id": "af",
        "sentencepiece_id": "af",
        "kenlm_id": "af",
    },
    {
        "lang": "Tosk Albanian",
        "oscar_id": "als",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "als",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Amharic",
        "oscar_id": "am",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "am",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Aragonese",
        "oscar_id": "an",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "an",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Arabic",
        "oscar_id": "ar",
        "stopwords_id": "ar",
        "badwords_id": "ar",
        "fasttext_id": "ar",
        "sentencepiece_id": "ar",
        "kenlm_id": "ar",
    },
    {
        "lang": "Egyptian Arabic",
        "oscar_id": "arz",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "arz",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Asturian",
        "oscar_id": "ast",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "ast",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Assamese",
        "oscar_id": "as",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "as",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Avaric",
        "oscar_id": "av",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "av",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "South Azerbaijani",
        "oscar_id": "azb",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "azb",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Azerbaijani",
        "oscar_id": "az",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "az",
        "sentencepiece_id": "az",
        "kenlm_id": "az",
    },
    {
        "lang": "Bavarian",
        "oscar_id": "bar",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "bar",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Bashkir",
        "oscar_id": "ba",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "ba",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Central Bikol",
        "oscar_id": "bcl",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "bcl",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Belarusian",
        "oscar_id": "be",
        "stopwords_id": None,
        "badwords_id": "be",
        "fasttext_id": "be",
        "sentencepiece_id": "be",
        "kenlm_id": "be",
    },
    {
        "lang": "Bulgarian",
        "oscar_id": "bg",
        "stopwords_id": "bg",
        "badwords_id": "bg",
        "fasttext_id": "bg",
        "sentencepiece_id": "bg",
        "kenlm_id": "bg",
    },
    {
        "lang": "Bihari",
        "oscar_id": "bh",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "bh",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Bengali",
        "oscar_id": "bn",
        "stopwords_id": "bn",
        "badwords_id": None,
        "fasttext_id": "bn",
        "sentencepiece_id": "bn",
        "kenlm_id": "bn",
    },
    {
        "lang": "Tibetan",
        "oscar_id": "bo",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "bo",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Bishnupriya",
        "oscar_id": "bpy",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "bpy",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Breton",
        "oscar_id": "br",
        "stopwords_id": "br",
        "badwords_id": None,
        "fasttext_id": "br",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Bosnian",
        "oscar_id": "bs",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "bs",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Russia Buriat",
        "oscar_id": "bxr",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "bxr",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Catalan",
        "oscar_id": "ca",
        "stopwords_id": "ca",
        "badwords_id": "ca",
        "fasttext_id": "ca",
        "sentencepiece_id": "ca",
        "kenlm_id": "ca",
    },
    {
        "lang": "Chavacano",
        "oscar_id": "cbk",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "cbk",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Cebuano",
        "oscar_id": "ceb",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "ceb",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Chechen",
        "oscar_id": "ce",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "ce",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Central Kurdish",
        "oscar_id": "ckb",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "ckb",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Czech",
        "oscar_id": "cs",
        "stopwords_id": "cs",
        "badwords_id": "cs",
        "fasttext_id": "cs",
        "sentencepiece_id": "cs",
        "kenlm_id": "cs",
    },
    {
        "lang": "Chuvash",
        "oscar_id": "cv",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "cv",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Welsh",
        "oscar_id": "cy",
        "stopwords_id": None,
        "badwords_id": "cy",
        "fasttext_id": "cy",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Danish",
        "oscar_id": "da",
        "stopwords_id": "da",
        "badwords_id": "da",
        "fasttext_id": "da",
        "sentencepiece_id": "da",
        "kenlm_id": "da",
    },
    {
        "lang": "German",
        "oscar_id": "de",
        "stopwords_id": "de",
        "badwords_id": "de",
        "fasttext_id": "de",
        "sentencepiece_id": "de",
        "kenlm_id": "de",
    },
    {
        "lang": "Dimli",
        "oscar_id": "diq",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "diq",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Lower Sorbian",
        "oscar_id": "dsb",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "dsb",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Dhivehi",
        "oscar_id": "dv",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "dv",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Modern Greek",
        "oscar_id": "el",
        "stopwords_id": "el",
        "badwords_id": "el",
        "fasttext_id": "el",
        "sentencepiece_id": "el",
        "kenlm_id": "el",
    },
    {
        "lang": "Emilian-Romagnol",
        "oscar_id": "eml",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "eml",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "English",
        "oscar_id": "en",
        "stopwords_id": "en",
        "badwords_id": "en",
        "fasttext_id": "en",
        "sentencepiece_id": "en",
        "kenlm_id": "en",
    },
    {
        "lang": "Esperanto",
        "oscar_id": "eo",
        "stopwords_id": "eo",
        "badwords_id": "eo",
        "fasttext_id": "eo",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Spanish",
        "oscar_id": "es",
        "stopwords_id": "es",
        "badwords_id": "es",
        "fasttext_id": "es",
        "sentencepiece_id": "es",
        "kenlm_id": "es",
    },
    {
        "lang": "Estonian",
        "oscar_id": "et",
        "stopwords_id": "et",
        "badwords_id": "et",
        "fasttext_id": "et",
        "sentencepiece_id": "et",
        "kenlm_id": "et",
    },
    {
        "lang": "Basque",
        "oscar_id": "eu",
        "stopwords_id": "eu",
        "badwords_id": "eu",
        "fasttext_id": "eu",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Persian",
        "oscar_id": "fa",
        "stopwords_id": "fa",
        "badwords_id": "fa",
        "fasttext_id": "fa",
        "sentencepiece_id": "fa",
        "kenlm_id": "fa",
    },
    {
        "lang": "Finnish",
        "oscar_id": "fi",
        "stopwords_id": "fi",
        "badwords_id": "fi",
        "fasttext_id": "fi",
        "sentencepiece_id": "fi",
        "kenlm_id": "fi",
    },
    {
        "lang": "Northern Frisian",
        "oscar_id": "frr",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "frr",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "French",
        "oscar_id": "fr",
        "stopwords_id": "fr",
        "badwords_id": "fr",
        "fasttext_id": "fr",
        "sentencepiece_id": "fr",
        "kenlm_id": "fr",
    },
    {
        "lang": "Western Frisian",
        "oscar_id": "fy",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "fy",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Irish",
        "oscar_id": "ga",
        "stopwords_id": "ga",
        "badwords_id": None,
        "fasttext_id": "ga",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Scottish Gaelic",
        "oscar_id": "gd",
        "stopwords_id": None,
        "badwords_id": "gd",
        "fasttext_id": "gd",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Galician",
        "oscar_id": "gl",
        "stopwords_id": "gl",
        "badwords_id": "gl",
        "fasttext_id": "gl",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Guarani",
        "oscar_id": "gn",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "gn",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Goan Konkani",
        "oscar_id": "gom",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "gom",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Gujarati",
        "oscar_id": "gu",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "gu",
        "sentencepiece_id": "gu",
        "kenlm_id": "gu",
    },
    {
        "lang": "Hebrew",
        "oscar_id": "he",
        "stopwords_id": "he",
        "badwords_id": None,
        "fasttext_id": "he",
        "sentencepiece_id": "he",
        "kenlm_id": "he",
    },
    {
        "lang": "Hindi",
        "oscar_id": "hi",
        "stopwords_id": "hi",
        "badwords_id": "hi",
        "fasttext_id": "hi",
        "sentencepiece_id": "hi",
        "kenlm_id": "hi",
    },
    {
        "lang": "Croatian",
        "oscar_id": "hr",
        "stopwords_id": "hr",
        "badwords_id": "hr",
        "fasttext_id": "hr",
        "sentencepiece_id": "hr",
        "kenlm_id": "hr",
    },
    {
        "lang": "Upper Sorbian",
        "oscar_id": "hsb",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "hsb",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Haitian",
        "oscar_id": "ht",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "ht",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Hungarian",
        "oscar_id": "hu",
        "stopwords_id": "hu",
        "badwords_id": "hu",
        "fasttext_id": "hu",
        "sentencepiece_id": "hu",
        "kenlm_id": "hu",
    },
    {
        "lang": "Armenian",
        "oscar_id": "hy",
        "stopwords_id": "hy",
        "badwords_id": "hy",
        "fasttext_id": "hy",
        "sentencepiece_id": "hy",
        "kenlm_id": "hy",
    },
    {
        "lang": "Interlingua",
        "oscar_id": "ia",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "ia",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Indonesian",
        "oscar_id": "id",
        "stopwords_id": "id",
        "badwords_id": "id",
        "fasttext_id": "id",
        "sentencepiece_id": "id",
        "kenlm_id": "id",
    },
    {
        "lang": "Interlingue",
        "oscar_id": "ie",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "ie",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Iloko",
        "oscar_id": "ilo",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "ilo",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Ido",
        "oscar_id": "io",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "io",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Icelandic",
        "oscar_id": "is",
        "stopwords_id": None,
        "badwords_id": "is",
        "fasttext_id": "is",
        "sentencepiece_id": "is",
        "kenlm_id": "is",
    },
    {
        "lang": "Italian",
        "oscar_id": "it",
        "stopwords_id": "it",
        "badwords_id": "it",
        "fasttext_id": "it",
        "sentencepiece_id": "it",
        "kenlm_id": "it",
    },
    {
        "lang": "Japanese",
        "oscar_id": "ja",
        "stopwords_id": "ja",
        "badwords_id": "ja",
        "fasttext_id": "ja",
        "sentencepiece_id": "ja",
        "kenlm_id": "ja",
    },
    {
        "lang": "Lojban",
        "oscar_id": "jbo",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "jbo",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Javanese",
        "oscar_id": "jv",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "jv",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Georgian",
        "oscar_id": "ka",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "ka",
        "sentencepiece_id": "ka",
        "kenlm_id": "ka",
    },
    {
        "lang": "Kazakh",
        "oscar_id": "kk",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "kk",
        "sentencepiece_id": "kk",
        "kenlm_id": "kk",
    },
    {
        "lang": "Central Khmer",
        "oscar_id": "km",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "km",
        "sentencepiece_id": "km",
        "kenlm_id": "km",
    },
    {
        "lang": "Kannada",
        "oscar_id": "kn",
        "stopwords_id": None,
        "badwords_id": "kn",
        "fasttext_id": "kn",
        "sentencepiece_id": "kn",
        "kenlm_id": "kn",
    },
    {
        "lang": "Korean",
        "oscar_id": "ko",
        "stopwords_id": "ko",
        "badwords_id": "ko",
        "fasttext_id": "ko",
        "sentencepiece_id": "ko",
        "kenlm_id": "ko",
    },
    {
        "lang": "Karachay-Balkar",
        "oscar_id": "krc",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "krc",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Kurdish",
        "oscar_id": "ku",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "ku",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Komi",
        "oscar_id": "kv",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "kv",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Cornish",
        "oscar_id": "kw",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "kw",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Kirghiz",
        "oscar_id": "ky",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "ky",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Latin",
        "oscar_id": "la",
        "stopwords_id": "la",
        "badwords_id": "la",
        "fasttext_id": "la",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Luxembourgish",
        "oscar_id": "lb",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "lb",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Lezghian",
        "oscar_id": "lez",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "lez",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Limburgan",
        "oscar_id": "li",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "li",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Lombard",
        "oscar_id": "lmo",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "lmo",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Lao",
        "oscar_id": "lo",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "lo",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Northern Luri",
        "oscar_id": "lrc",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "lrc",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Lithuanian",
        "oscar_id": "lt",
        "stopwords_id": None,
        "badwords_id": "lt",
        "fasttext_id": "lt",
        "sentencepiece_id": "lt",
        "kenlm_id": "lt",
    },
    {
        "lang": "Latvian",
        "oscar_id": "lv",
        "stopwords_id": "lv",
        "badwords_id": "lv",
        "fasttext_id": "lv",
        "sentencepiece_id": "lv",
        "kenlm_id": "lv",
    },
    {
        "lang": "Maithili",
        "oscar_id": "mai",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "mai",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Malagasy",
        "oscar_id": "mg",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "mg",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Eastern Mari",
        "oscar_id": "mhr",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "mhr",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Minangkabau",
        "oscar_id": "min",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "min",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Macedonian",
        "oscar_id": "mk",
        "stopwords_id": None,
        "badwords_id": "mk",
        "fasttext_id": "mk",
        "sentencepiece_id": "mk",
        "kenlm_id": "mk",
    },
    {
        "lang": "Malayalam",
        "oscar_id": "ml",
        "stopwords_id": None,
        "badwords_id": "ml",
        "fasttext_id": "ml",
        "sentencepiece_id": "ml",
        "kenlm_id": "ml",
    },
    {
        "lang": "Mongolian",
        "oscar_id": "mn",
        "stopwords_id": None,
        "badwords_id": "mn",
        "fasttext_id": "mn",
        "sentencepiece_id": "mn",
        "kenlm_id": "mn",
    },
    {
        "lang": "Western Mari",
        "oscar_id": "mrj",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "mrj",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Marathi",
        "oscar_id": "mr",
        "stopwords_id": "mr",
        "badwords_id": "mr",
        "fasttext_id": "mr",
        "sentencepiece_id": "mr",
        "kenlm_id": "mr",
    },
    {
        "lang": "Malay",
        "oscar_id": "ms",
        "stopwords_id": None,
        "badwords_id": "ms",
        "fasttext_id": "ms",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Maltese",
        "oscar_id": "mt",
        "stopwords_id": None,
        "badwords_id": "mt",
        "fasttext_id": "mt",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Mirandese",
        "oscar_id": "mwl",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "mwl",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Burmese",
        "oscar_id": "my",
        "stopwords_id": None,
        "badwords_id": "my",
        "fasttext_id": "my",
        "sentencepiece_id": "my",
        "kenlm_id": "my",
    },
    {
        "lang": "Erzya",
        "oscar_id": "myv",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "myv",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Mazanderani",
        "oscar_id": "mzn",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "mzn",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Nahuatl",
        "oscar_id": "nah",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "nah",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Neapolitan",
        "oscar_id": "nap",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "nap",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Low German",
        "oscar_id": "nds",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "nds",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Nepali",
        "oscar_id": "ne",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "ne",
        "sentencepiece_id": "ne",
        "kenlm_id": "ne",
    },
    {
        "lang": "Newari",
        "oscar_id": "new",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "new",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Dutch",
        "oscar_id": "nl",
        "stopwords_id": "nl",
        "badwords_id": "nl",
        "fasttext_id": "nl",
        "sentencepiece_id": "nl",
        "kenlm_id": "nl",
    },
    {
        "lang": "Norwegian Nynorsk",
        "oscar_id": "nn",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "nn",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Norwegian",
        "oscar_id": "no",
        "stopwords_id": "no",
        "badwords_id": "no",
        "fasttext_id": "no",
        "sentencepiece_id": "no",
        "kenlm_id": "no",
    },
    {
        "lang": "Occitan",
        "oscar_id": "oc",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "oc",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Oriya",
        "oscar_id": "or",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "or",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Ossetian",
        "oscar_id": "os",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "os",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Pampanga",
        "oscar_id": "pam",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "pam",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Panjabi",
        "oscar_id": "pa",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "pa",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Polish",
        "oscar_id": "pl",
        "stopwords_id": "pl",
        "badwords_id": "pl",
        "fasttext_id": "pl",
        "sentencepiece_id": "pl",
        "kenlm_id": "pl",
    },
    {
        "lang": "Piemontese",
        "oscar_id": "pms",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "pms",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Western Panjabi",
        "oscar_id": "pnb",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "pnb",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Pushto",
        "oscar_id": "ps",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "ps",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Portuguese",
        "oscar_id": "pt",
        "stopwords_id": "pt",
        "badwords_id": "pt",
        "fasttext_id": "pt",
        "sentencepiece_id": "pt",
        "kenlm_id": "pt",
    },
    {
        "lang": "Quechua",
        "oscar_id": "qu",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "qu",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Romansh",
        "oscar_id": "rm",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "rm",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Romanian",
        "oscar_id": "ro",
        "stopwords_id": "ro",
        "badwords_id": "ro",
        "fasttext_id": "ro",
        "sentencepiece_id": "ro",
        "kenlm_id": "ro",
    },
    {
        "lang": "Russian",
        "oscar_id": "ru",
        "stopwords_id": "ru",
        "badwords_id": "ru",
        "fasttext_id": "ru",
        "sentencepiece_id": "ru",
        "kenlm_id": "ru",
    },
    {
        "lang": "Yakut",
        "oscar_id": "sah",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "sah",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Sanskrit",
        "oscar_id": "sa",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "sa",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Sicilian",
        "oscar_id": "scn",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "scn",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Sindhi",
        "oscar_id": "sd",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "sd",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Serbo-Croatian",
        "oscar_id": "sh",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "sh",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Sinhala",
        "oscar_id": "si",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "si",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Slovak",
        "oscar_id": "sk",
        "stopwords_id": "sk",
        "badwords_id": "sk",
        "fasttext_id": "sk",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Slovenian",
        "oscar_id": "sl",
        "stopwords_id": "sl",
        "badwords_id": "sl",
        "fasttext_id": "sl",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Somali",
        "oscar_id": "so",
        "stopwords_id": "so",
        "badwords_id": None,
        "fasttext_id": "so",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Albanian",
        "oscar_id": "sq",
        "stopwords_id": None,
        "badwords_id": "sq",
        "fasttext_id": "sq",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Serbian",
        "oscar_id": "sr",
        "stopwords_id": None,
        "badwords_id": "sr",
        "fasttext_id": "sr",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Sundanese",
        "oscar_id": "su",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "su",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Swedish",
        "oscar_id": "sv",
        "stopwords_id": "sv",
        "badwords_id": "sv",
        "fasttext_id": "sv",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Swahili",
        "oscar_id": "sw",
        "stopwords_id": "sw",
        "badwords_id": None,
        "fasttext_id": "sw",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Tamil",
        "oscar_id": "ta",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "ta",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Telugu",
        "oscar_id": "te",
        "stopwords_id": None,
        "badwords_id": "te",
        "fasttext_id": "te",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Tajik",
        "oscar_id": "tg",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "tg",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Thai",
        "oscar_id": "th",
        "stopwords_id": "th",
        "badwords_id": "th",
        "fasttext_id": "th",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Turkmen",
        "oscar_id": "tk",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "tk",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Tagalog",
        "oscar_id": "tl",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "tl",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Turkish",
        "oscar_id": "tr",
        "stopwords_id": "tr",
        "badwords_id": "tr",
        "fasttext_id": "tr",
        "sentencepiece_id": "tr",
        "kenlm_id": "tr",
    },
    {
        "lang": "Tatar",
        "oscar_id": "tt",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "tt",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Tuvinian",
        "oscar_id": "tyv",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "tyv",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Uighur",
        "oscar_id": "ug",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "ug",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Ukrainian",
        "oscar_id": "uk",
        "stopwords_id": None,
        "badwords_id": "uk",
        "fasttext_id": "uk",
        "sentencepiece_id": "uk",
        "kenlm_id": "uk",
    },
    {
        "lang": "Urdu",
        "oscar_id": "ur",
        "stopwords_id": "ur",
        "badwords_id": None,
        "fasttext_id": "ur",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Uzbek",
        "oscar_id": "uz",
        "stopwords_id": None,
        "badwords_id": "uz",
        "fasttext_id": "uz",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Venetian",
        "oscar_id": "vec",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "vec",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Vietnamese",
        "oscar_id": "vi",
        "stopwords_id": "vi",
        "badwords_id": "vi",
        "fasttext_id": "vi",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Volapük",
        "oscar_id": "vo",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "vo",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Waray",
        "oscar_id": "war",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "war",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Walloon",
        "oscar_id": "wa",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "wa",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Wu Chinese",
        "oscar_id": "wuu",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "wuu",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Kalmyk",
        "oscar_id": "xal",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "xal",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Mingrelian",
        "oscar_id": "xmf",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "xmf",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Yiddish",
        "oscar_id": "yi",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "yi",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Yoruba",
        "oscar_id": "yo",
        "stopwords_id": "yo",
        "badwords_id": None,
        "fasttext_id": "yo",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Yue Chinese",
        "oscar_id": "yue",
        "stopwords_id": None,
        "badwords_id": None,
        "fasttext_id": "yue",
        "sentencepiece_id": None,
        "kenlm_id": None,
    },
    {
        "lang": "Chinese",
        "oscar_id": "zh",
        "stopwords_id": "zh",
        "badwords_id": "zh",
        "fasttext_id": "zh",
        "sentencepiece_id": "zh",
        "kenlm_id": "zh",
    },
]
langs_id = pd.DataFrame(langs_id)
