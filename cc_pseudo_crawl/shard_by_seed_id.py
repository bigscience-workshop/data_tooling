import logging
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

from datasets import Dataset, load_from_disk, concatenate_datasets
from datasets.utils.logging import set_verbosity_info

"""
Deduplicating using `datasets` is much harder, we but we forgot to generate an id when building an index, so we're screwed.
"""

set_verbosity_info()
logger = logging.getLogger(__name__)

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True, help="Dataset directory containing all shards.")
    parser.add_argument("--save-dir", type=str, required=True, help="Where to save the datasets.")
    parser.add_argument("--num-proc", type=int, default=1, help="Number of procs use for preprocessing.")
    args = parser.parse_args()

    args.dataset_dir = Path(args.dataset_dir)
    args.save_dir = Path(args.save_dir)
    return args

def obtain_entire_dataset(dataset_dir: Path) -> Dataset:
    shard_paths = dataset_dir.iterdir()
    shards = [load_from_disk(str(shard_path.absolute())) for shard_path in shard_paths]
    return concatenate_datasets(shards)

def shard_by_seed_id(ds: Dataset, num_proc: int) -> Dict[int, Dataset]:
    seed_ids = set(ds["seed_id"])
    result = {}

    for seed_id in seed_ids:
        result[seed_id] = ds.filter(lambda row: row["seed_id"] == seed_id, num_proc=num_proc, batched=True)

    return result

def deduplicate_url(ds: Dataset) -> Dataset:
    url_to_timestamp_and_index = {}
    for id_, row in enumerate(ds):
        url = row["url"]
        timestamp = row["fetch_time"]
        if url not in url_to_timestamp_and_index:
            url_to_timestamp_and_index[url] = (timestamp, id_)
            continue

        previous_timestamp, previous_id = url_to_timestamp_and_index[url]
        if timestamp > previous_timestamp:
            url_to_timestamp_and_index[url] = (timestamp, id_)

    # Deduplicate url
    indices_to_keep = [id_ for _, id_ in url_to_timestamp_and_index.values()]
    return ds.select(indices_to_keep)

def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = get_args()
    logger.info(
        f"** The job is runned with the following arguments: **\n{args}\n **** "
    )

    # Concatenate all the shards together
    ds = obtain_entire_dataset(args.dataset_dir)

    # Deduplicate url
    ds = deduplicate_url(ds)

    # Split dataset according to seed_id
    shards = shard_by_seed_id(ds, args.num_proc)

    # Save shard per seed
    for seed_id, shard in shards.items():
        save_path = args.save_dir / f"seed_id={seed_id}"
        shard.save_to_disk(f"{str(save_path.absolute())}.tmp")
        subprocess.run(
            ["mv", f"{str(save_path.absolute())}.tmp", str(save_path.absolute())]
        )

if __name__ == "__main__":
    main()
