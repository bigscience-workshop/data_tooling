import logging
import os
import subprocess
from argparse import ArgumentParser
from math import ceil
from pathlib import Path
from typing import List, Dict

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
    return ceil(ds_nbytes / max_size)

def shard_dataset(ds: Dataset, max_size: int) -> List[Dataset]:
    """The idea is to shard everything in order for final shards to be 10G of less"""

    number_shards = compute_number_of_shards(ds, max_size)

    if number_shards <= 1:
        return [ds]
    results = []
    logger.info(f"Shard dataset in {number_shards} shards")
    for shard_id in range(number_shards):
        logger.info(f"Shard {shard_id}/{number_shards}")
        shard = ds.shard(num_shards=number_shards, index=shard_id)
        results.append(shard)
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

    selected_mime_types = {"text/html"}
    splits: Dict[str, Dataset] = {
        **{
            mime_type: ds.filter(
                lambda mime_types_: [mime_type_ == mime_type for mime_type_ in mime_types_],
                input_columns="content_mime_detected", batched=True, num_proc=args.num_proc
            )
            for mime_type in selected_mime_types
        },
        "others": ds.filter(
            lambda mime_types_: [mime_type_ not in selected_mime_types for mime_type_ in mime_types_],
            input_columns="content_mime_detected", batched=True, num_proc=args.num_proc
        )
    }
    shards: Dict[str, List[Dataset]] = {
        key: shard_dataset(split, args.max_size)
        for key, split in splits.items()
    }
    for key, shards_per_split in shards.items():
        folder_name = key.replace("/", "__")
        save_split_path: Path = args.save_path / folder_name
        save_split_path.mkdir(parents=True, exist_ok=True)
        num_shards = len(shards_per_split)
        for i, shard_per_split in enumerate(shards_per_split):
            if key == "text/html":
                shard_per_split = shard_per_split.remove_columns("compressed_warc")
                save_path = save_split_path / f"shard-id-{i}--{num_shards}.jsonl.gz"
                shard_per_split.to_json(
                    f"{str(save_path.absolute())}.tmp",
                    num_proc=args.num_proc,
                    compression="gzip"
                )
            else:
                save_path = save_split_path / f"shard-id-{i}--{num_shards}"
                shard_per_split.save_to_disk(
                    f"{str(save_path.absolute())}.tmp",
                )
            subprocess.run(
                ["mv", f"{str(save_path.absolute())}.tmp", str(save_path.absolute())]
            )

if __name__ == "__main__":
    main()