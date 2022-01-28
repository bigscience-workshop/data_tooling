import logging
from argparse import ArgumentParser
from math import ceil
from pathlib import Path
from typing import List

from datasets import Dataset, load_from_disk
from datasets.utils.logging import set_verbosity_info

set_verbosity_info()
logger = logging.getLogger(__name__)

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True, help="Dataset path.")
    parser.add_argument("--max-size", type=int, required=True, help="Max shards sizes.")
    parser.add_argument("--save-path", type=str, required=True, help="Where to save the dataset.")
    parser.add_argument("--num-proc", type=int, default=1, help="Number of procs use for preprocessing.")
    args = parser.parse_args()

    args.dataset_path = Path(args.dataset_path)
    args.save_path = Path(args.save_path)
    return args

def compute_number_of_shards(ds: Dataset, max_size: int) -> int:
    if ds._indices is not None:
        ds_nbytes = ds.data.nbytes * len(ds._indices) / len(ds.data)
    else:
        ds_nbytes = ds.data.nbytes

    logger.info(f"Estimated dataset size: {ds_nbytes} bytes")
    logger.info(f"Max shard size: {max_size} bytes")
    return ceil(max_size / ds_nbytes)

def load_and_shard_dataset(ds: Dataset, max_size: int) -> List[Dataset]:
    """The idea is to shard everything in order for final shards to be 10G of less"""

    number_shards = compute_number_of_shards(ds, max_size)

    if number_shards <= 1:
        return [ds]

    results = []
    logger.info(f"Shard dataset in {number_shards} shards")
    for shard_id in range(number_shards):
        logger.info(f"Shard {shard_id}/{number_shards}")
        results.append(ds.shard(num_shards=number_shards, index=shard_id))
    return results

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

    ds = load_from_disk(str(args.dataset_path.absolute()))
    shards = load_and_shard_dataset(ds, args.max_size)
    num_shards = len(shards)
    for i, shard in enumerate(shards):
        shard.to_json(args.save_path / f"shard-id-{i}--{num_shards}" , num_proc=args.num_proc, compression="gzip")

if __name__ == "__main__":
    main()