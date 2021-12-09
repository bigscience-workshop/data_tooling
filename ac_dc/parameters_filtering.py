import string
import emoji


main_special_characters = string.punctuation + string.digits + string.whitespace
other_special_characters = (
    "    　    ￼’“”–ー一▬…✦�­£​•€«»°·═"
    "×士＾˘⇓↓↑←→（）§″′´¿−±∈﻿¢ø‚„½¼¾¹²³―⁃，ˌ¸‹›ʺˈʻ¦‐⠀‰‑≤≥‖"
    "◆●■►▼▲▴∆▻¡★☆✱ːº。¯˜¥ɪ≈†上ン：∼⁄・♡✓⊕․．⋅÷１‟；،、¨ाাी्े◦˚"
    "゜ʼ≖ʼ¤ッツシ℃√！【】‿∞➤～πه۩☛₨➩☻๑٪♥ıॽ《‘©﴿٬x？▷Г♫∟™ª₪®「—"
    "❖」﴾》"
)
emoji = list(emoji.UNICODE_EMOJI["en"].keys())

special_characters_default = set(main_special_characters + other_special_characters)
special_characters_default.update(emoji)


parameters_filtering_default = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": False,
    "length_word_max_cutoff": 50,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.4,
    "cond_check_stopwords": False,
    "stopwords_min_cutoff": 0,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.70,
    "cond_check_perplexity": False,
    "perplexity_max_cutoff": 3000000,
}

parameters_filtering_af = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 25,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.3,
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.6,
    "cond_check_perplexity": True,
    "perplexity_max_cutoff": 3000000,
}

parameters_filtering_ar = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 25,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.45,
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.75,
    "cond_check_perplexity": True,
    "perplexity_max_cutoff": 1000000,
}

parameters_filtering_arz = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 25,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.5,
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.75,
    "cond_check_perplexity": False,
    "perplexity_max_cutoff": 3000000,
}

parameters_filtering_as = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 25,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.25,
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.75,
    "cond_check_perplexity": False,
    "perplexity_max_cutoff": 3000000,
}

parameters_filtering_bn = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 30,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.275,
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0.05,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.75,
    "cond_check_perplexity": False,
    "perplexity_max_cutoff": 575000,
}

parameters_filtering_ca = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 30,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.35,
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.75,
    "cond_check_perplexity": True,
    "perplexity_max_cutoff": 1750000,
}

parameters_filtering_en = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": True,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 25,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 20,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.4,
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0.3,
    "cond_check_badwords": True,
    "badwords_max_cutoff": 0.045,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.80,
    "cond_check_perplexity": True,
    "perplexity_max_cutoff": 2500,
}

parameters_filtering_es = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 30,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.3,
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0.2,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.75,
    "cond_check_perplexity": True,
    "perplexity_max_cutoff": 2500000,
}

parameters_filtering_eu = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 35,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.3,
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.75,
    "cond_check_perplexity": False,
    "perplexity_max_cutoff": 3000000,
}

parameters_filtering_fr = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 30,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.35,
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0.15,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.75,
    "cond_check_perplexity": True,
    "perplexity_max_cutoff": 3000000,
}

parameters_filtering_gu = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 30,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.3,
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.75,
    "cond_check_perplexity": True,
    "perplexity_max_cutoff": 250000,
}

parameters_filtering_hi = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 25,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.35,
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.75,
    "cond_check_perplexity": True,
    "perplexity_max_cutoff": 600000,
}

parameters_filtering_id = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 30,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.25,
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0.25,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.75,
    "cond_check_perplexity": True,
    "perplexity_max_cutoff": 2500000,
}

parameters_filtering_kn = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 50,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.25,
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.75,
    "cond_check_perplexity": True,
    "perplexity_max_cutoff": 400000,
}

parameters_filtering_ml = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 50,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.2,
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.75,
    "cond_check_perplexity": True,
    "perplexity_max_cutoff": 1600000,
}

parameters_filtering_mr = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 30,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.25,
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.75,
    "cond_check_perplexity": True,
    "perplexity_max_cutoff": 425000,
}

parameters_filtering_pt = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 30,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.3,
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0.15,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.75,
    "cond_check_perplexity": True,
    "perplexity_max_cutoff": 3000000,
}

parameters_filtering_so = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": False,
    "length_word_max_cutoff": 1000,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.3,
    "cond_check_stopwords": False,
    "stopwords_min_cutoff": 0,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.75,
    "cond_check_perplexity": False,
    "perplexity_max_cutoff": 3000000,
}

parameters_filtering_sw = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 30,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.275,
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.75,
    "cond_check_perplexity": False,
    "perplexity_max_cutoff": 3000000,
}

parameters_filtering_ta = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 50,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.25,
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.75,
    "cond_check_perplexity": False,
    "perplexity_max_cutoff": 3000000,
}

parameters_filtering_te = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 35,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.25,
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.75,
    "cond_check_perplexity": False,
    "perplexity_max_cutoff": 3000000,
}

parameters_filtering_ur = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 30,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.4,
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.75,
    "cond_check_perplexity": False,
    "perplexity_max_cutoff": 3000000,
}

parameters_filtering_vi = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 30,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.35,
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.75,
    "cond_check_perplexity": False,
    "perplexity_max_cutoff": 3000000,
}

parameters_filtering_yo = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 30,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.3,
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.75,
    "cond_check_perplexity": False,
    "perplexity_max_cutoff": 3000000,
}

parameters_filtering_zh = {
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": False,
    "length_word_max_cutoff": 1000,
    "cond_check_number_words": True,
    "number_words_min_cutoff": 1,
    "number_words_max_cutoff": 100000,
    "strip_characters": special_characters_default,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.4,
    "cond_check_stopwords": False,
    "stopwords_min_cutoff": 0,
    "cond_check_badwords": False,
    "badwords_max_cutoff": 0.2,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.75,
    "cond_check_perplexity": False,
    "perplexity_max_cutoff": 3000000,
}

parameters_filtering = {
    "default": parameters_filtering_default,
    "af": parameters_filtering_af,
    "ar": parameters_filtering_ar,
    "arz": parameters_filtering_arz,
    "as": parameters_filtering_as,
    "bn": parameters_filtering_bn,
    "ca": parameters_filtering_ca,
    "en": parameters_filtering_en,
    "es": parameters_filtering_es,
    "eu": parameters_filtering_eu,
    "fr": parameters_filtering_fr,
    "gu": parameters_filtering_gu,
    "hi": parameters_filtering_hi,
    "id": parameters_filtering_id,
    "kn": parameters_filtering_kn,
    "ml": parameters_filtering_ml,
    "mr": parameters_filtering_mr,
    "pt": parameters_filtering_pt,
    "so": parameters_filtering_so,
    "sw": parameters_filtering_sw,
    "ta": parameters_filtering_ta,
    "te": parameters_filtering_te,
    "ur": parameters_filtering_ur,
    "vi": parameters_filtering_vi,
    "yo": parameters_filtering_yo,
    "zh": parameters_filtering_zh,
}
