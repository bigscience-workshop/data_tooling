import os
import urllib.request

import kenlm


class KenlmModel:
    def __init__(self, language):
        download_kenlm_model(language)
        try:
            self.model = kenlm.Model(f"{language}.arpa.bin")
        except OSError:
            os.remove(f"{language}.arpa.bin")
            if os.path.exists(f"{language}.sp.model"):
                os.remove(f"{language}.sp.model")
            raise OSError(
                "File was corrupt and should have been removed. Please, retry."
            )

    @classmethod
    def from_pretrained(cls, language: str):
        return cls(language)

    def get_perplexity(self, doc: str):
        doc_log_score, doc_length = 0, 0
        for line in doc.split("\n"):
            log_score = self.model.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length
        return 10.0 ** (-doc_log_score / doc_length)


def download_kenlm_model(language: str):
    root_url = "http://dl.fbaipublicfiles.com/cc_net/lm"
    bin_name = f"{language}.arpa.bin"
    model_name = f"{language}.sp.model"
    bin_url = f"{root_url}/{bin_name}"
    model_url = f"{root_url}/{model_name}"

    if not os.path.isfile(bin_name):
        urllib.request.urlretrieve(bin_url, bin_name)

    if not os.path.isfile(model_name):
        urllib.request.urlretrieve(model_url, model_name)
