def lower_strip_sentence(sentence):
    sent = sentence.lower().strip()
    return sent


def get_words_from_sentence(sentence, strip_characters):
    """Get lower case words from a sentence."""
    sent = lower_strip_sentence(sentence)
    words = [word.strip(strip_characters) for word in sent.split(" ")]
    return words


def check_empty(sentence, strip_characters):
    """Return True if sentence is not empty."""
    sent = lower_strip_sentence(sentence)
    words = get_words_from_sentence(sentence, strip_characters)
    cond = (len(sent) > 0) and (len(words) > 0)
    return cond


def check_special_characters(
    sentence,
    special_characters,
    special_characters_cutoff,
):
    """
    Return True if the ratio of special characters in the sentence is below the cutoff.
    """
    sent = lower_strip_sentence(sentence)
    set_special_characters = {char for char in special_characters}
    special_characters_ratio = len(
        [char for char in sent if char in set_special_characters]
    ) / len(sent)
    cond = special_characters_ratio < special_characters_cutoff
    return cond


def check_stopwords(
    sentence,
    strip_characters,
    stopwords,
    stopwords_cutoff,
):
    """
    Return True if the ratio of stopwords in the sentence is below the cutoff.
    """
    cond = True
    if stopwords:
        words = get_words_from_sentence(sentence, strip_characters)
        stopwords_ratio = len([word for word in words if word in stopwords]) / len(
            words
        )
        cond = stopwords_ratio < stopwords_cutoff
    return cond


def check_badwords(
    sentence,
    strip_characters,
    badwords,
    badwords_cutoff,
):
    cond = True
    if badwords:
        words = get_words_from_sentence(sentence, strip_characters)
        badwords_ratio = len([word for word in words if word in badwords]) / len(words)
        cond = badwords_ratio < badwords_cutoff
    return cond


def check_lang_id(
    sentence,
    strip_characters,
    lang_oscar_id,
    model_lang_id,
    lang_id_cutoff,
    fasttext_lang_id,
):
    cond = True
    if model_lang_id:
        words = get_words_from_sentence(sentence, strip_characters)
        sent = " ".join(words)
        pred = model_lang_id.predict(sent)
        lang_pred_fasttext_id = pred[0][0].replace("__label__", "")
        score_pred = pred[1][0]
        cond = (fasttext_lang_id == lang_oscar_id) and (score_pred > lang_id_cutoff)

    return cond
