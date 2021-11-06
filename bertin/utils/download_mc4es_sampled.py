import gzip
import io
import json
import sys

import requests
from tqdm import tqdm

_DATA_URL_TRAIN = "https://huggingface.co/datasets/bertin-project/mc4-es-sampled/resolve/main/mc4-es-train-50M-{config}-shard-{index:04d}-of-{n_shards:04d}.json.gz"


def main(config="stepwise"):
    data_urls = [
        _DATA_URL_TRAIN.format(
            config=config,
            index=index + 1,
            n_shards=1024,
        )
        for index in range(1024)
    ]
    with open(f"mc4-es-train-50M-{config}.jsonl", "w") as f:
        for dara_url in tqdm(data_urls):
            response = requests.get(dara_url)
            bio = io.BytesIO(response.content)
            with gzip.open(bio, "rt", encoding="utf8") as g:
                for line in g:
                    json_line = json.loads(line.strip())
                    f.write(json.dumps(json_line) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
