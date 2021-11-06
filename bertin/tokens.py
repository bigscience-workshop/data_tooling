#!/usr/bin/env python3
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

# Load dataset
dataset = load_dataset("oscar", "unshuffled_deduplicated_es", split="train[:5000000]")

# Instantiate tokenizer
tokenizer = ByteLevelBPETokenizer()


def batch_iterator(batch_size=100_000):
    for i in range(0, len(dataset), batch_size):
        yield dataset["text"][i : i + batch_size]


# Customized training
tokenizer.train_from_iterator(
    batch_iterator(),
    vocab_size=50265,
    min_frequency=2,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ],
)
# Save files to disk
tokenizer.save("./tokenizer.json")
