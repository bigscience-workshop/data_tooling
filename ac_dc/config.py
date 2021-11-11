from logging import getLogger


logger = getLogger(__name__)


class PipelineConfig:
    """Base class for all pipeline configurations

    Class attributes (overridden by derived classes)

        - language_code: Two character language code (ISO 3166) for the pipeline

    Parameters for filters:

        - strip_characters": "' 0123456789¯_%$§½¼¾×|†~\"—±′–'°−{}[]·'?,./<>!@#^&*()+-‑=:;，：,...�`→¶.…‘’”"
        - cond_check_empty": True
        - cond_check_special_characters": True
        - special_characters": "' 0123456789¯_%$§½¼¾×|†~\"—±′–'°−{}[]·'?,./<>!@#^&*()+-‑=:;，：,...�`→¶.…‘’”"
        - special_characters_cutoff": 0.4
        - cond_check_stopwords": True
        - stopwords_cutoff": 0.4
        - cond_check_badwords": True
        - badwords_cutoff": 0.4
        - cond_check_lang_id": True
        - lang_id_cutoff": 0.8


    Parameters for normalers:

        - cond_remove_words_with_incorrect_substrings": True,
        - incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
        - cond_remove_long_words": False,
        - length_word_cutoff": 50,

    """

    attribute_map = {}

    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)

    def __init__(self, language_code, **kwargs):
        self.language_code = language_code
        self.strip_character = kwargs.get("strip_characters", "")
        self.cond_check_empty = kwargs.get("cond_check_empty", True)
        self.cond_check_special_characters = kwargs.get(
            "cond_check_special_characters", True
        )
        self.special_characters = kwargs.get("special_characters", "")
        self.special_characters_cutoff = kwargs.get("special_characters_cutoff", 0.4)
        self.cond_check_stopwords = kwargs.get("cond_check_stopwords", True)
        self.stopwords_cutoff = kwargs.get("stopwords_cutoff", 0.4)
        self.cond_check_badwords = kwargs.get("cond_check_badwords", True)
        self.badwords_cutoff = kwargs.get("badwords_cutoff", 0.4)
        self.cond_check_lang_id = kwargs.get("cond_check_lang_id", True)
        self.lang_id_cutoff = kwargs.get("lang_id_cutoff", 0.8)
        self.cond_remove_words_with_incorrect_substrings = kwargs.get(
            "cond_remove_words_with_incorrect_substrings", True
        )

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err
