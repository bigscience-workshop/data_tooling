#!/usr/bin/env python
import kenlm
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

TOTAL_SENTENCES = 20000


def pp(log_score, length):
    return 10.0 ** (-log_score / length)


embedder = "distiluse-base-multilingual-cased-v1"
embedder_model = SentenceTransformer(embedder)
embedding_shape = embedder_model.encode(["foo"])[0].shape[0]
# http://dl.fbaipublicfiles.com/cc_net/lm/es.arpa.bin
model = kenlm.Model("es.arpa.bin")
mc4 = load_dataset("mc4", "es", streaming=True)
count = 0
embeddings = []
lenghts = []
perplexities = []
sentences = []

for sample in tqdm(mc4["train"].shuffle(buffer_size=100_000), total=416057992):
    lines = sample["text"].split("\n")
    for line in lines:
        count += 1
        log_score = model.score(line)
        length = len(line.split()) + 1
        embedding = embedder_model.encode([line])[0]
        embeddings.append(embedding.tolist())
        perplexities.append(pp(log_score, length))
        lenghts.append(length)
        sentences.append(line)
        if count == TOTAL_SENTENCES:
            break
    if count == TOTAL_SENTENCES:
        embeddings = np.array(embeddings)
        df = pd.DataFrame(
            {"sentence": sentences, "length": lenghts, "perplexity": perplexities}
        )
        for dim in range(embedding_shape):
            df[f"dim_{dim}"] = embeddings[:, dim]
        df.to_csv("mc4-es-perplexity-sentences.tsv", index=None, sep="\t")
        print("DONE!")
        break
