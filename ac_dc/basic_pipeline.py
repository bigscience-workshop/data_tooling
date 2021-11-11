import fasttext
from . import filters, normalizers
from .pipeline import BasePipeline
from .config import PipelineConfig
from .badwords import badwords


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
                self.config.model_lang_id,
                self.config.lang_id_cutoff,
            ):
                return False
        return True


if __name__ == "__main__":
    lang_oscar_id = "en"
    path_model_fasttext = "/tmp/lid.176.bin"
    path_oscar_file = "../en_part_1.txt.gz"

    from tqdm import tqdm
    import gzip
    import os

    parameters_filtering_en = {
        "cond_remove_words_with_incorrect_substrings": True,
        "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
        "cond_remove_long_words": True,
        "length_word_cutoff": 25,
        "cond_check_empty": True,
        "strip_characters": "' 0123456789¯_%$§½¼¾×|†~\"—±′–'°−{}[]·'?,./<>!@#^&*()+-‑=:;，：,...�`→¶.…‘’”",
        "cond_check_special_characters": True,
        "special_characters": "' 0123456789¯_%$§½¼¾×|†~\"—±′–'°−{}[]·'?,./<>!@#^&*()+-‑=:;，：,...�`→¶.…‘’”",
        "special_characters_cutoff": 0.4,
        "cond_check_stopwords": True,
        "stopwords_cutoff": 0.4,
        "cond_check_badwords": True,
        "badwords_cutoff": 0.4,
        "badwords": badwords[lang_oscar_id],
        "cond_check_lang_id": True,
        "path_model_fasttext": path_model_fasttext,
        "lang_id_cutoff": 0.8,
    }
    config = PipelineConfig("en", **parameters_filtering_en)
    pipeline = BasicPipeline(config)

    with open(
        path_oscar_file.replace(".txt.gz", "") + ".sample_filtered.txt",
        "w",
        encoding="utf8",
    ) as f:
        with gzip.open(path_oscar_file, "rb") as f2:
            from tqdm import tqdm as tqdm

            for id_, line in enumerate(tqdm(f2)):
                line = pipeline.normalize(line.decode().strip())
                if pipeline.filter(line):
                    f.write(line + "\n\n")

        os.unlink(path_oscar_file)
