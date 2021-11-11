def remove_words_with_incorrect_substrings(
    sentence,
    incorrect_word_substrings,
):
    words = sentence.split(" ")
    words = [
        word
        for word in words
        if all([(i_substr not in word) for i_substr in incorrect_word_substrings])
    ]
    filtered_sentence = " ".join(words)
    return filtered_sentence


def remove_long_words(
    sentence,
    length_word_cutoff,
):
    words = sentence.split(" ")
    words = [word for word in words if len(word) < length_word_cutoff]
    filtered_sentence = " ".join(words)
    return filtered_sentence
