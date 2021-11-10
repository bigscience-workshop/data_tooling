parameters_filtering_en = {
    "cond_remove_words_with_incorrect_substrings": True,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_cutoff": 25,
    "cond_check_empty": True,
    "strip_characters": "' 0123456789¯_%$§½¼¾×|†~\"—±′–'°−{}[]·\'?,./<>!@#^&*()+-‑=:;，：,...�`→¶.…‘’”",
    "cond_check_special_characters": True,
    "special_characters": "' 0123456789¯_%$§½¼¾×|†~\"—±′–'°−{}[]·\'?,./<>!@#^&*()+-‑=:;，：,...�`→¶.…‘’”",
    "special_characters_cutoff": 0.4,
    "cond_check_stopwords": True,
    "stopwords_cutoff": 0.4,
    "cond_check_badwords": False,
    "badwords_cutoff": 0.4,
    "cond_check_lang_id": True,
    "lang_id_cutoff": 0.8,
}

parameters_filtering = {
    "en": parameters_filtering_en,
}
