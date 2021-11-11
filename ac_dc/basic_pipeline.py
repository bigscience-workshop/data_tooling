import fasttext
from . import filters, normalizers
from .pipeline import BasePipeline
from .config import PipelineConfig


class BasicPipeline(BasePipeline):
    """A Basic pipeline to start."""

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        if self.config.cond_check_lang_id:
            self.model_lang_id = fasttext.load_model(config.path_model_fasttext)

    def normalize(self, sentence: str) -> str:
        if self.config.cond_remove_words_with_incorrect_substrings:
            sentence = normalizers.remove_words_with_incorrect_substrings(
                sentence,
                self.config.incorrect_word_substrings,
            )

        if self.config.cond_remove_long_words:
            sentence = normalizers.remove_long_words(
                sentence,
                self.config.length_word_cutoff,
            )

        return sentence

    def filter(self, sentence: str) -> bool:
        if self.config.cond_check_empty:
            if not filters.check_empty(sentence, self.config.strip_characters):
                return False

        if self.config.cond_check_special_characters:
            if not filters.check_special_characters(
                sentence,
                self.config.special_characters,
                self.config.special_characters_cutoff,
            ):
                return False
        if self.config.cond_check_stopwords:
            if not filters.check_stopwords(
                sentence,
                self.config.strip_characters,
                self.config.stopwords,
                self.config.stopwords_cutoff,
            ):
                return False
        if self.config.cond_check_badwords:
            if not filters.check_badwords(
                sentence,
                self.config.strip_characters,
                self.config.badwords,
                self.config.badwords_cutoff,
            ):
                return False
        if self.config.cond_check_lang_id:
            if not filters.check_lang_id(
                sentence,
                self.config.strip_characters,
                self.config.lang_oscar_id,
                self.model_lang_id,
                self.config.lang_id_cutoff,
            ):
                return False
        return True
