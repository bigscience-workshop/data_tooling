import json
import logging
import os

from datasets import load_dataset
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level="INFO",
    datefmt="[%X]",
)

# Log on each process the small summary:
logger = logging.getLogger(__name__)
os.system("wget http://dl.fbaipublicfiles.com/cc_net/lm/es.arpa.bin")
mc4 = load_dataset(
    "./mc4",
    "es",
    split="train",
    sampling_method="steps",
    perplexity_model="./es.arpa.bin",
    sampling_factor=1.5e5,
    boundaries=[536394.99320948, 662247.50212365, 919250.87225178],
    streaming=True,
).shuffle(buffer_size=10000, seed=2021)
total = 0
with open("mc4-es-train-50M-steps.jsonl", "w") as f:
    for sample in tqdm(mc4, total=50_000_000):
        f.write(json.dumps(sample) + "\n")
        total += 1
        if total >= 50_000_000:
            break

mc4val = load_dataset(
    "./mc4",
    "es",
    split="validation",
    sampling_method="steps",
    perplexity_model="./es.arpa.bin",
    sampling_factor=5e5,
    boundaries=[536394.99320948, 662247.50212365, 919250.87225178],
    streaming=True,
).shuffle(buffer_size=10000, seed=2021)
total = 0
with open("mc4-es-validation-5M-steps.jsonl", "w") as f:
    for sample in tqdm(mc4val, total=5_000_000):
        f.write(json.dumps(sample) + "\n")
        total += 1
        if total >= 5_000_000:
            break


# ------------------

import json
import logging

from datasets import load_dataset
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level="INFO",
    datefmt="[%X]",
)

# Log on each process the small summary:
logger = logging.getLogger(__name__)


mc4 = load_dataset(
    "./mc4",
    "es",
    split="train",
    sampling_method="gaussian",
    perplexity_model="../es.arpa.bin",
    sampling_factor=0.78,
    boundaries=[536394.99320948, 662247.50212365, 919250.87225178],
    streaming=True,
).shuffle(buffer_size=10000, seed=2021)
total = 0
with open("mc4-es-train-50M-gaussian.jsonl", "w") as f:
    for sample in tqdm(mc4, total=50_000_000):
        f.write(json.dumps(sample) + "\n")
        total += 1
        if total >= 50_000_000:
            break
mc4val = load_dataset(
    "./mc4",
    "es",
    split="validation",
    sampling_method="gaussian",
    perplexity_model="../es.arpa.bin",
    sampling_factor=1,
    boundaries=[536394.99320948, 662247.50212365, 919250.87225178],
    streaming=True,
).shuffle(buffer_size=10000, seed=2021)
total = 0
with open("mc4-es-validation-5M-gaussian.jsonl", "w") as f:
    for sample in tqdm(mc4val, total=5_000_000):
        f.write(json.dumps(sample) + "\n")
        total += 1
        if total >= 5_000_000:
            break


# ------------------

import json
import logging

from datasets import load_dataset
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level="INFO",
    datefmt="[%X]",
)

# Log on each process the small summary:
logger = logging.getLogger(__name__)


mc4 = load_dataset(
    "./mc4",
    "es",
    split="train",
    sampling_method="random",
    perplexity_model="../es.arpa.bin",
    sampling_factor=0.5,
    boundaries=[536394.99320948, 662247.50212365, 919250.87225178],
    streaming=True,
).shuffle(buffer_size=10000, seed=2021)
total = 0
with open("mc4-es-train-50M-random.jsonl", "w") as f:
    for sample in tqdm(mc4, total=50_000_000):
        f.write(json.dumps(sample) + "\n")
        total += 1
        if total >= 50_000_000:
            break
mc4val = load_dataset(
    "./mc4",
    "es",
    split="validation",
    sampling_method="random",
    perplexity_model="../es.arpa.bin",
    sampling_factor=0.5,
    boundaries=[536394.99320948, 662247.50212365, 919250.87225178],
    streaming=True,
).shuffle(buffer_size=10000, seed=2021)
total = 0
with open("mc4-es-validation-5M-random.jsonl", "w") as f:
    for sample in tqdm(mc4val, total=5_000_000):
        f.write(json.dumps(sample) + "\n")
        total += 1
        if total >= 5_000_000:
            break
