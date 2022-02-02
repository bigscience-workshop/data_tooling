import ast
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

from datasets import load_dataset, Dataset, concatenate_datasets

set_verbosity_info()
logger = logging.getLogger(__name__)


def parse_dataset_with_ratio(elt: str):
    tuple_ = ast.literal_eval(elt)
    assert len(tuple_) == 2
    result = (tuple_[0], float(tuple_[1]))
    assert result[1] <= 1 and result[1] >= 0, "Ratio should be between 0 and 1"
    return result

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--datasets-with_ratio",
        type=lambda x: [parse_dataset_with_ratio(elt) for elt in x.split(",")],
        required=True
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        required=True
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        required=True
    )

    args = parser.parse_args()

    return args

def collapse_meta_(batch):
    """{"text": str, "meta": str}"""
    # TODO: check that
    columns_not_in_meta = ["text", "html_str"]
    columns_to_collapse = [name for name in batch.keys() if name not in columns_not_in_meta]

    new_batch = {
        "text": batch["text"],
        "meta": [
            str({key: value for key, value in zip(columns_to_collapse, row)})
            for row in zip(*[batch[name] for name in columns_to_collapse])
        ]
    }
    return new_batch

def collapse_meta(ds: Dataset, num_proc):
    """{"text": str, "meta": str}"""
    columns_to_keep = ["text"]
    column_names_to_remove = [name for name in ds.column_names if name not in columns_to_keep]
    return ds.map(collapse_meta_, batched=True, num_proc=num_proc, remove_columns=column_names_to_remove)

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

    sampled_datasets: Dict[str, Dataset] = {}
    for dataset_path, ratio in args.datasets_with_ratio:
        ds = load_dataset(dataset_path)

        # collapse all meta data in "meta" column
        ds = collapse_meta(ds, args.num_proc)

        # randomly sample ratio * len(ds)
        # TODO: build more efficiently
        ds = ds.shuffle(seed=42)
        ds = ds.select(range(int(ratio * len(ds))))

        sampled_datasets[dataset_path] = ds
        logger.info(f"Processed {dataset_path}")

    ds = concatenate_datasets(list(sampled_datasets.values()))

    # Save dataset locally
    logger.info(f"Save dataset at {args.save_path}")
    save_path_tmp = Path(f"{str(args.save_path.absolute())}.tmp")
    ds.save_to_disk(str(save_path_tmp.absolute()))
    save_path_tmp.rename(args.save_path)

if __name__ == "__main__":
    main()