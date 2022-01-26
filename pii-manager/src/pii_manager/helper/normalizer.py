def normalize(
    text: str, lang: str, whitespace: bool = True, lowercase: bool = False
) -> str:
    """
    Perforn some normalization steps on a text string
    """
    if whitespace:
        text = " ".join(text.split())

    if lowercase:
        text = text.lower()

    return text
