import os
import json
import logging
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from statistics import mean

import datasets
from datasets import config, load_from_disk
from datasets.utils.logging import set_verbosity_info

set_verbosity_info()
logger = logging.getLogger(__name__)


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="path to the parquet dataset folder",
    )
    parser.add_argument(
        "--save-path-stats-json",
        type=str,
        required=True,
        help="Where to save the stats json.",
    )
    parser.add_argument(
        "--save-path-stats-full-json", type=str, help="Where to save the stats json."
    )
    parser.add_argument(
        "--save-batch-size", type=int, required=True, help="Batch size when writing."
    )
    parser.add_argument("--use-datasets-caching", action="store_true")
    parser.add_argument(
        "--num-proc", type=int, default=1, help="Number of procs use for preprocessing."
    )
    parser.add_argument(
        "--seed-id",
        type=int,
        required=True,
        help="Value of the seed id.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Optional argument to select a subset (used for debugging purposes). Example `10`.",
    )
    args = parser.parse_args()

    return args


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

    if os.path.isfile(args.save_path_stats_json):
        logger.info(f" --- Statistics already computed for seed id {args.seed_id} ")
        return

    logger.info(f" --- Statistics not already computed for seed id {args.seed_id} ")
    if not args.use_datasets_caching:
        datasets.set_caching_enabled(False)
    else:
        logger.info(
            f"the datasets results will be cached at {config.HF_DATASETS_CACHE}."
        )

    ds = load_from_disk(args.dataset_path)

    if args.num_examples is not None:
        ds = ds.select([i for i in range(args.num_examples)])

    selected_mime_types = ["text/html"]
    splits = {
        **{
            mime_type: ds.filter(
                lambda mime_types_: [
                    mime_type_ == mime_type for mime_type_ in mime_types_
                ],
                input_columns="content_mime_detected",
                batched=True,
                num_proc=args.num_proc,
            )
            for mime_type in selected_mime_types
        },
        "others": ds.filter(
            lambda mime_types_: [
                mime_type_ not in selected_mime_types for mime_type_ in mime_types_
            ],
            input_columns="content_mime_detected",
            batched=True,
            num_proc=args.num_proc,
        ),
    }

    data_stats = {f"{split_name}_total": len(ds) for split_name, ds in splits.items()}

    ds_html = splits[selected_mime_types[0]]

    logger.info(f"the currents splits are {data_stats}.")

    def get_length_text(example):
        example["length_text"] = (
            len(example["text"]) if example["text"] is not None else 0
        )
        return example

    cols_to_remove = [
        col
        for col in ds.column_names
        if col not in ["content_languages", "url_host_tld"]
    ]
    ds_html = ds_html.map(
        get_length_text,
        batched=False,
        num_proc=args.num_proc,
        remove_columns=cols_to_remove,
    )

    data_stats["html_empty_text"] = len([e for e in ds_html["length_text"] if e == 0])

    non_empty_texts = [e for e in ds_html["length_text"] if e != 0]
    data_stats["html_mean_length_non_empty_text"] = (
        mean(non_empty_texts) if non_empty_texts != [] else None
    )
    data_stats["seed_id"] = args.seed_id

    logger.info(
        f"There is {data_stats['html_empty_text']} empty text rows out of {len(ds_html)} rows."
    )

    save_path = Path(args.save_path_stats_json)
    save_path_tmp = f"{str(save_path.absolute())}.tmp"
    logger.info(f"Saving the dataset at {save_path_tmp}")
    with open(save_path_tmp, "w", encoding="utf-8") as f:
        json.dump(data_stats, f, ensure_ascii=False, indent=4)
    logger.info(f"Moving the saved dataset to {str(save_path.absolute())}")
    subprocess.run(["mv", save_path_tmp, str(save_path.absolute())])

    save_path = Path(args.save_path_stats_full_json)
    tmp_file_name = f"tmp-{str(save_path.name)}"
    save_path_tmp = os.path.join(save_path.parent, tmp_file_name)
    logger.info(f"Saving the dataset at {save_path_tmp}")
    ds_html.to_json(
        save_path_tmp,
        batch_size=args.save_batch_size,
        num_proc=args.num_proc,
        compression="gzip",
    )
    logger.info(f"Moving the saved dataset to {str(save_path.absolute())}")
    subprocess.run(["mv", save_path_tmp, str(save_path.absolute())])


if __name__ == "__main__":
    main()
