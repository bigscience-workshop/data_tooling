import json

import kenlm
from tqdm import tqdm

model = kenlm.Model("../es.arpa.bin")


def get_perplexity(doc):
    doc_log_score, doc_length = 0, 0
    for line in doc.split("\n"):
        log_score = model.score(line)
        length = len(line.split()) + 1
        doc_log_score += log_score
        doc_length += length
    return 10.0 ** (-doc_log_score / doc_length)


with open("mc4-es-train-50M-stats.csv", "w") as csv:
    with open("mc4-es-train-50M-steps.jsonl", "r") as data:
        for line in tqdm(data):
            text = json.loads(line)["text"]
            csv.write(f"{len(text.split())},{get_perplexity(text)}\n")
