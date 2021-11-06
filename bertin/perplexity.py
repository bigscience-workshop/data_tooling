#!/usr/bin/env python
import kenlm
from datasets import load_dataset
from tqdm import tqdm


def pp(log_score, length):
    return 10.0 ** (-log_score / length)


# http://dl.fbaipublicfiles.com/cc_net/lm/es.arpa.bin
model = kenlm.Model("es.arpa.bin")
mc4 = load_dataset("mc4", "es", streaming=True)
with open("mc4-es-perplexity.txt", "w") as f:
    for sample in tqdm(mc4["train"].shuffle(buffer_size=100_000), total=416057992):
        lines = sample["text"].split("\n")
        doc_log_score, doc_length = 0, 0
        for line in lines:
            log_score = model.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length
        f.write(f"{pp(doc_log_score, doc_length)}\n")
