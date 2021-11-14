special_characters_default = (
    " ~!@#$%^&*{}[]()_+=-0987654321`<>,./?':;“”\"\t\n\\πه☆●¦″"
    "．۩۱（☛₨➩°・■↑☻、๑º‹€σ٪’Ø·−♥ıॽ،٥《‘©。¨﴿！★×✱´٬→±x：¹？£―▷ф"
    "¡Г♫∟™ª₪®▬「—¯；¼❖․ø•�」٣，٢◦‑←§١ー٤）˚›٩▼٠«¢¸٨³½˜٭ˈ¿¬ι۞⌐¥►"
    "†ƒ∙²»¤…﴾⠀》′ا✓"
)

parameters_filtering_default = {
    "cond_remove_words_with_incorrect_substrings": True,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": False,
    "length_word_cutoff": 50,
    "cond_check_empty": True,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_cutoff": 0.4,
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": -0.01,
    "stopwords_max_cutoff": 0.60,
    "cond_check_badwords": True,
    "badwords_cutoff": 0.4,
    "cond_check_lang_id": True,
    "lang_id_cutoff": 0.8,
    "cond_check_perplexity": False,
    "perplexity_cutoff": 1000000,
}

parameters_filtering_en = parameters_filtering_default
parameters_filtering_en["cond_remove_long_words"] = True
parameters_filtering_en["length_word_cutoff"] = 25

parameters_filtering = {
    "default": parameters_filtering_default,
    "en": parameters_filtering_en,
}
