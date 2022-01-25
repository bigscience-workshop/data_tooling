import logging
import subprocess
from argparse import ArgumentParser
from multiprocessing import Pool
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
    parser.add_argument("--dataset-path", type=str, required=True, help="Dataset path.")
    parser.add_argument("--save-prefix-path", type=str, required=True, help="Where to save the dataset.")
    parser.add_argument("--num-proc", type=int, default=1, help="Number of procs use for preprocessing.")
    args = parser.parse_args()

    args.dataset_path = Path(args.dataset_path)
    args.save_prefix_path = Path(args.save_prefix_path)
    return args

def obtain_entire_dataset(dataset_dir: Path, num_proc: int) -> Dataset:
    shard_paths = sorted([str(elt.absolute()) for elt in dataset_dir.iterdir()])
    logger.info(f"All the following shards will be loaded: {shard_paths}")
    # shards = []
    # for shard_path in shard_paths:
    #     logger.info(f"Loading {shard_path}")
    #     shard = load_from_disk(shard_path)
    #     shards.append(shard)
    # Parallel version seems to go OOM
    with Pool(num_proc) as pool:
        async_results = pool.map_async(load_from_disk, shard_paths)
        shards = async_results.get()
    logger.info("Concatenating all shards together.")
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

def run_on_entire_dataset(args):
    # Concatenate all the shards together
    logger.info("Concatenating all datasets together")
    ds = obtain_entire_dataset(args.dataset_dir, 10)

    # Filter some generic things
    logger.info("Filtering bad seeds")

    def filter_func(batch):
        # 508: https://zh.wikipedia.org/
        # 577: https://fr.wikipedia.org/
        # 523: https://github.com/
        # 529: https://sites.google.com/
        bad_seeds = [508, 523, 529, 577]
        return [row["seed_id"] not in bad_seeds for row in batch]

    ds = ds.filter(filter_func, batched=True, num_proc=args.num_proc)

    # Deduplicate url
    logger.info("Deduplicating urls")
    ds = deduplicate_url(ds)

    # Split dataset according to seed_id
    logger.info("Sharding by seed id")
    shards = shard_by_seed_id(ds, args.num_proc)

    # Save shard per seed
    logger.info("Saving shards")
    for seed_id, shard in shards.items():
        save_path = args.save_dir / f"seed_id={seed_id}"
        shard.save_to_disk(f"{str(save_path.absolute())}.tmp")
        subprocess.run(
            ["mv", f"{str(save_path.absolute())}.tmp", str(save_path.absolute())]
        )

def run_on_shard(args):
    ds = load_from_disk(args.dataset_path)

    # Filter some generic things
    logger.info("Filtering bad seeds")
    def filter_func(batch):
        # 508: https://zh.wikipedia.org/
        # 577: https://fr.wikipedia.org/
        # 523: https://github.com/
        # 529: https://sites.google.com/
        bad_seeds = [508, 523, 529, 577]
        return [row["seed_id"] not in bad_seeds for row in batch]
    ds = ds.filter(filter_func, batched=True, num_proc=args.num_proc)

    # This has to be done at some point.
    # # Deduplicate url
    # logger.info("Deduplicating urls")
    # ds = deduplicate_url(ds)

    # Split dataset according to seed_id
    logger.info("Sharding by seed id")
    shards = shard_by_seed_id(ds, args.num_proc)

    # Save shard per seed
    logger.info("Saving shards")
    for seed_id, shard in shards.items():
        save_path = f"{str(args.save_prefix_path.absolute())}--seed_id--{seed_id}"
        shard.save_to_disk(f"{save_path}.tmp")
        subprocess.run(
            ["mv", f"{save_path}.tmp", save_path]
        )


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

    run_on_shard(args)

if __name__ == "__main__":
    main()
