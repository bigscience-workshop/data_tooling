import re
import unicodedata
from typing import Dict

import kenlm
import sentencepiece


class SentencePiece:
    def __init__(
        self,
        model: str,
    ):
        super().__init__()
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.load(str(model))

    def do(self, text: str) -> str:
        tokenized = self.sp.encode_as_pieces(text)
        return " ".join(tokenized)


class KenlmModel:
    digit_re: re.Pattern = re.compile(r"\d")
    unicode_punct: Dict[str, str] = {
        "，": ",",
        "。": ".",
        "、": ",",
        "„": '"',
        "”": '"',
        "“": '"',
        "«": '"',
        "»": '"',
        "１": '"',
        "」": '"',
        "「": '"',
        "《": '"',
        "》": '"',
        "´": "'",
        "∶": ":",
        "：": ":",
        "？": "?",
        "！": "!",
        "（": "(",
        "）": ")",
        "；": ";",
        "–": "-",
        "—": " - ",
        "．": ". ",
        "～": "~",
        "’": "'",
        "…": "...",
        "━": "-",
        "〈": "<",
        "〉": ">",
        "【": "[",
        "】": "]",
        "％": "%",
        "►": "-",
    }
    unicode_punct_re = re.compile(f"[{''.join(unicode_punct.keys())}]")
    non_printing_chars_re = re.compile(
        f"[{''.join(map(chr, list(range(0,32)) + list(range(127,160))))}]"
    )

    def __init__(self, kenlm_model: str, sentence_piece_model: str):
        self.model = kenlm.Model(kenlm_model)
        self.tokenizer = SentencePiece(sentence_piece_model)

    def pp(self, log_score, length):
        return 10.0 ** (-log_score / length)

    def get_perplexity(self, doc: str, normalize_cc_net: bool = True):
        if normalize_cc_net:
            doc = self.normalize(doc)
        # Tokenize (after normalizing): See https://github.com/facebookresearch/cc_net/blob/bda555bd1cf1ee2e0b925363e62a61cd46c8b60d/cc_net/mine.py#L352 for full pipeline
        doc = self.tokenizer.do(doc)
        doc_log_score, doc_length = 0, 0
        for line in doc.split("\n"):
            log_score = self.model.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length
        return round(self.pp(doc_log_score, doc_length), 1)

    def normalize(
        self,
        line: str,
        accent: bool = True,
        case: bool = True,
        numbers: bool = True,
        punct: int = 1,
    ) -> str:
        line = line.strip()
        if not line:
            return line
        if case:
            line = line.lower()
        if accent:
            line = self.strip_accents(line)
        if numbers:
            line = self.digit_re.sub("0", line)
        if punct == 1:
            line = self.replace_unicode_punct(line)
        elif punct == 2:
            line = self.remove_unicode_punct(line)
        line = self.remove_non_printing_char(line)
        return line

    def strip_accents(self, line: str) -> str:
        """Strips accents from a piece of text."""
        nfd = unicodedata.normalize("NFD", line)
        output = [c for c in nfd if unicodedata.category(c) != "Mn"]
        if len(output) == line:
            return line
        return "".join(output)

    def replace_unicode_punct(self, text: str) -> str:
        return "".join(self.unicode_punct.get(c, c) for c in text)

    def remove_unicode_punct(self, text: str) -> str:
        """More aggressive version of replace_unicode_punct but also faster."""
        return self.unicode_punct_re.sub("", text)

    def remove_non_printing_char(self, text: str) -> str:
        return self.non_printing_chars_re.sub("", text)
