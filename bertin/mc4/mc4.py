"""Perplexity Sampled mC4 dataset based on Common Crawl."""


import gzip
import json

import datasets
import kenlm  # pip install https://github.com/kpu/kenlm/archive/master.zip
import numpy as np
from numpy.random import default_rng

logger = datasets.logging.get_logger(__name__)


_DESCRIPTION = """\
A colossal, cleaned version of Common Crawl's web crawl corpus.

Based on Common Crawl dataset: "https://commoncrawl.org".

This is the processed version of Google's mC4 dataset by AllenAI.
"""

_CITATION = """
@article{2019t5,
    author = {Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu},
    title = {Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
    journal = {arXiv e-prints},
    year = {2019},
    archivePrefix = {arXiv},
    eprint = {1910.10683},
}
"""

_URL = "https://github.com/allenai/allennlp/discussions/5056"

_DATA_URL = "https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/multilingual/c4-{language}{split_suffix}.tfrecord-{index:05d}-of-{n_shards:05d}.json.gz"

_LANGUAGES = [
    "af",
    "am",
    "ar",
    "az",
    "be",
    "bg",
    "bg-Latn",
    "bn",
    "ca",
    "ceb",
    "co",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "el-Latn",
    "en",
    "eo",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fil",
    "fr",
    "fy",
    "ga",
    "gd",
    "gl",
    "gu",
    "ha",
    "haw",
    "hi",
    "hi-Latn",
    "hmn",
    "ht",
    "hu",
    "hy",
    "id",
    "ig",
    "is",
    "it",
    "iw",
    "ja",
    "ja-Latn",
    "jv",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "ku",
    "ky",
    "la",
    "lb",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "no",
    "ny",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "ru-Latn",
    "sd",
    "si",
    "sk",
    "sl",
    "sm",
    "sn",
    "so",
    "sq",
    "sr",
    "st",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tr",
    "uk",
    "und",
    "ur",
    "uz",
    "vi",
    "xh",
    "yi",
    "yo",
    "zh",
    "zh-Latn",
    "zu",
]

_N_SHARDS_PER_SPLIT = {
    "af": {"train": 64, "validation": 1},
    "am": {"train": 16, "validation": 1},
    "ar": {"train": 1024, "validation": 4},
    "az": {"train": 256, "validation": 1},
    "be": {"train": 128, "validation": 1},
    "bg": {"train": 1024, "validation": 1},
    "bg-Latn": {"train": 4, "validation": 1},
    "bn": {"train": 512, "validation": 1},
    "ca": {"train": 512, "validation": 1},
    "ceb": {"train": 8, "validation": 1},
    "co": {"train": 8, "validation": 1},
    "cs": {"train": 1024, "validation": 2},
    "cy": {"train": 256, "validation": 1},
    "da": {"train": 1024, "validation": 1},
    "de": {"train": 2048, "validation": 16},
    "el": {"train": 1024, "validation": 2},
    "el-Latn": {"train": 16, "validation": 1},
    "en": {"train": 11264, "validation": 128},
    "eo": {"train": 32, "validation": 1},
    "es": {"train": 2048, "validation": 16},
    "et": {"train": 256, "validation": 1},
    "eu": {"train": 64, "validation": 1},
    "fa": {"train": 1024, "validation": 2},
    "fi": {"train": 1024, "validation": 1},
    "fil": {"train": 64, "validation": 1},
    "fr": {"train": 2048, "validation": 16},
    "fy": {"train": 16, "validation": 1},
    "ga": {"train": 16, "validation": 1},
    "gd": {"train": 16, "validation": 1},
    "gl": {"train": 128, "validation": 1},
    "gu": {"train": 64, "validation": 1},
    "ha": {"train": 8, "validation": 1},
    "haw": {"train": 2, "validation": 1},
    "hi": {"train": 1024, "validation": 2},
    "hi-Latn": {"train": 16, "validation": 1},
    "hmn": {"train": 8, "validation": 1},
    "ht": {"train": 8, "validation": 1},
    "hu": {"train": 1024, "validation": 2},
    "hy": {"train": 128, "validation": 1},
    "id": {"train": 1024, "validation": 4},
    "ig": {"train": 4, "validation": 1},
    "is": {"train": 128, "validation": 1},
    "it": {"train": 1024, "validation": 8},
    "iw": {"train": 1024, "validation": 1},
    "ja": {"train": 1024, "validation": 8},
    "ja-Latn": {"train": 8, "validation": 1},
    "jv": {"train": 8, "validation": 1},
    "ka": {"train": 256, "validation": 1},
    "kk": {"train": 256, "validation": 1},
    "km": {"train": 64, "validation": 1},
    "kn": {"train": 64, "validation": 1},
    "ko": {"train": 1024, "validation": 1},
    "ku": {"train": 16, "validation": 1},
    "ky": {"train": 64, "validation": 1},
    "la": {"train": 64, "validation": 1},
    "lb": {"train": 32, "validation": 1},
    "lo": {"train": 8, "validation": 1},
    "lt": {"train": 512, "validation": 1},
    "lv": {"train": 256, "validation": 1},
    "mg": {"train": 8, "validation": 1},
    "mi": {"train": 4, "validation": 1},
    "mk": {"train": 128, "validation": 1},
    "ml": {"train": 128, "validation": 1},
    "mn": {"train": 128, "validation": 1},
    "mr": {"train": 1024, "validation": 1},
    "ms": {"train": 512, "validation": 1},
    "mt": {"train": 128, "validation": 1},
    "my": {"train": 64, "validation": 1},
    "ne": {"train": 256, "validation": 1},
    "nl": {"train": 1024, "validation": 4},
    "no": {"train": 1024, "validation": 1},
    "ny": {"train": 4, "validation": 1},
    "pa": {"train": 32, "validation": 1},
    "pl": {"train": 1024, "validation": 4},
    "ps": {"train": 16, "validation": 1},
    "pt": {"train": 1024, "validation": 4},
    "ro": {"train": 1024, "validation": 2},
    "ru": {"train": 4096, "validation": 32},
    "ru-Latn": {"train": 32, "validation": 1},
    "sd": {"train": 64, "validation": 1},
    "si": {"train": 64, "validation": 1},
    "sk": {"train": 512, "validation": 1},
    "sl": {"train": 256, "validation": 1},
    "sm": {"train": 4, "validation": 1},
    "sn": {"train": 8, "validation": 1},
    "so": {"train": 64, "validation": 1},
    "sq": {"train": 128, "validation": 1},
    "sr": {"train": 256, "validation": 1},
    "st": {"train": 2, "validation": 1},
    "su": {"train": 4, "validation": 1},
    "sv": {"train": 1024, "validation": 2},
    "sw": {"train": 32, "validation": 1},
    "ta": {"train": 256, "validation": 1},
    "te": {"train": 128, "validation": 1},
    "tg": {"train": 64, "validation": 1},
    "th": {"train": 1024, "validation": 1},
    "tr": {"train": 1024, "validation": 4},
    "uk": {"train": 1024, "validation": 2},
    "und": {"train": 3072, "validation": 32},
    "ur": {"train": 128, "validation": 1},
    "uz": {"train": 32, "validation": 1},
    "vi": {"train": 1024, "validation": 4},
    "xh": {"train": 2, "validation": 1},
    "yi": {"train": 16, "validation": 1},
    "yo": {"train": 2, "validation": 1},
    "zh": {"train": 1024, "validation": 2},
    "zh-Latn": {"train": 8, "validation": 1},
    "zu": {"train": 8, "validation": 1},
}


class Mc4Config(datasets.BuilderConfig):
    """BuilderConfig for mC4."""

    def __init__(self, *args, languages, **kwargs):
        """BuilderConfig for mC4.
        Args:
            languages (:obj:`List[str]`): list of languages to load
            **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(
            *args,
            name="+".join(languages),
            **kwargs,
        )
        self.languages = languages


class Mc4(datasets.GeneratorBasedBuilder):
    """mC4, a colossal, cleaned version of Common Crawl's web crawl corpus."""

    BUILDER_CONFIGS = [Mc4Config(languages=[lang]) for lang in _LANGUAGES]
    BUILDER_CONFIG_CLASS = Mc4Config

    def __init__(self, *args, writer_batch_size=None, **kwargs):
        self.data_files = kwargs.pop("data_files", {})
        self.sampling_method = kwargs.pop("sampling_method", None)
        self.perplexity_model = kwargs.pop("perplexity_model", None)
        self.sampling_factor = kwargs.pop("sampling_factor", None)
        self.boundaries = kwargs.pop("boundaries", None)
        self.seed = kwargs.pop("seed", None)
        self.kwargs = kwargs
        if self.sampling_method:
            if self.seed is not None:
                self.rng = default_rng(self.seed)
            else:
                self.rng = default_rng()
            if self.sampling_method == "random":
                self.should_keep_doc = self._should_keep_doc_random
            else:
                # Loading 5-gram model
                # http://dl.fbaipublicfiles.com/cc_net/lm/es.arpa.bin
                logger.info("loading model = %s", self.perplexity_model)
                self.pp_model = kenlm.Model(self.perplexity_model)
                if self.sampling_method == "gaussian":
                    self.should_keep_doc = self._should_keep_doc_gaussian
                else:
                    self.should_keep_doc = self._should_keep_doc_step
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)

    def get_perplexity(self, doc):
        doc_log_score, doc_length = 0, 0
        for line in doc.split("\n"):
            log_score = self.pp_model.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length
        return 10.0 ** (-doc_log_score / doc_length)

    def _should_keep_doc_step(self, doc, factor=1.5e5, boundaries=None, **kwargs):
        perplexity = self.get_perplexity(doc)
        if boundaries is None:
            boundaries = [536394.99320948, 662247.50212365, 919250.87225178]
        if perplexity <= boundaries[0]:
            quartile_range = boundaries[0]
        elif boundaries[0] < perplexity < boundaries[1]:
            quartile_range = boundaries[1] - boundaries[0]
        elif boundaries[1] < perplexity < boundaries[2]:
            quartile_range = boundaries[2] - boundaries[1]
        elif perplexity >= boundaries[2]:
            quartile_range = 10 * boundaries[2]
        probability = factor / quartile_range
        return self.rng.uniform() < probability

    def _should_keep_doc_gaussian(self, doc, factor=0.78, boundaries=None, **kwargs):
        width = kwargs.get("width", 9 / 2)  # width (spread) of the exponential curve
        perplexity = self.get_perplexity(doc)
        if boundaries is not None:
            m = boundaries[1]
        else:
            m = 662247.50212365
        exponential = np.exp((-1 / width) * ((perplexity - m) / m) ** 2)
        weighted_perplexity = factor * exponential
        return self.rng.uniform() < weighted_perplexity

    def _should_keep_doc_random(self, doc, factor=None, boundaries=None, **kwargs):
        if factor is None:
            factor = 0.5
        return self.rng.uniform() <= factor

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "timestamp": datasets.Value("string"),
                    "url": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_urls = {}
        for split in ["train", "validation"]:
            data_urls[split] = [
                _DATA_URL.format(
                    language=self.config.name,
                    split_suffix="-validation" if split == "validation" else "",
                    index=index,
                    n_shards=_N_SHARDS_PER_SPLIT[lang][split],
                )
                for lang in self.config.languages
                for index in range(_N_SHARDS_PER_SPLIT[lang][split])
            ]
        if self.data_files and "train" in self.data_files:
            train_downloaded_files = self.data_files["train"]
            if not isinstance(train_downloaded_files, (tuple, list)):
                train_downloaded_files = [train_downloaded_files]
        else:
            train_downloaded_files = dl_manager.download(data_urls["train"])
        if self.data_files and "validation" in self.data_files:
            validation_downloaded_files = self.data_files["validation"]
            if not isinstance(validation_downloaded_files, (tuple, list)):
                validation_downloaded_files = [validation_downloaded_files]
        else:
            validation_downloaded_files = dl_manager.download(data_urls["validation"])
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepaths": train_downloaded_files},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepaths": validation_downloaded_files},
            ),
        ]

    def _generate_examples(self, filepaths):
        """This function returns the examples in the raw (text) form by iterating on all the files."""
        id_ = 0
        for filepath in filepaths:
            logger.info("generating examples from = %s", filepath)
            if filepath.endswith("jsonl"):
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line:
                            example = json.loads(line)
                            yield id_, example
                            id_ += 1
            else:
                with gzip.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
                    if self.sampling_method:
                        logger.info("sampling method = %s", self.sampling_method)
                        for line in f:
                            if line:
                                example = json.loads(line)
                                if self.should_keep_doc(
                                    example["text"],
                                    factor=self.sampling_factor,
                                    boundaries=self.boundaries,
                                    **self.kwargs,
                                ):
                                    yield id_, example
                                    id_ += 1
                    else:
                        for line in f:
                            if line:
                                example = json.loads(line)
                                yield id_, example
                                id_ += 1
