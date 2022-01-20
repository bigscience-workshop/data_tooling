import json
import logging
import re
import subprocess
import threading
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

import boto3
import botocore
import datasets
from botocore.config import Config
from botocore.exceptions import ClientError
from datasets import config, load_dataset
from datasets.utils.logging import set_verbosity_info

"""
Required: obtain cc_index and copy it locally
`aws s3 sync s3://commoncrawl-dev/big-science-workshop/data-sourcing-sheet/cc-{FLAVOR}/ $CC_INDEX_FOLDER/cc-{FLAVOR}/`
"""

set_verbosity_info()
logger = logging.getLogger(__name__)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument(
        "--cc-index-folder",
        type=str,
        required=True,
        help="Folder containing index dataset in parquet format.",
    )
    parser.add_argument(
        "--save-dir", type=str, required=True, help="Where to save the datasets."
    )
    parser.add_argument(
        "--num-proc", type=int, default=1, help="Number of procs use for preprocessing."
    )
    parser.add_argument(
        "--range",
        type=str,
        default=None,
        help="Optional argument to select a subset (used for debugging purposes). Example `:10`.",
    )
    parser.add_argument("--shard-id", type=int, help="Preprocess dataset via shards.")
    parser.add_argument("--num-shards", type=int, help="Total number of shards.")
    parser.add_argument("--use-datasets-caching", action="store_true")

    args = parser.parse_args()

    matches = re.match(
        r"^bigscience-catalogue-data/pseudo_crawl_(?:(.*)_partial|(seed))(_dedup_url)?$",
        args.dataset,
    )
    assert matches is not None
    flavors = [elt for elt in matches.groups() if elt is not None]
    assert len(flavors) == 1 or (len(flavors) == 2 and flavors[1] == "_dedup_url")
    flavor = flavors[0]
    assert (
        flavor == "seed"
        or re.match(r"^intermediate_depth_([0-9]+)$", flavor) is not None
    )
    args.cc_index_folder = Path(args.cc_index_folder) / f"cc-{''.join(flavors)}"
    args.flavor = flavor

    if args.shard_id is not None:
        assert args.num_shards is not None

    return args


thread_data = threading.local()


def set_global_session():
    if not hasattr(thread_data, "s3_client"):
        thread_data.s3_client = boto3.session.Session(region_name="us-east-1").client(
            "s3", config=Config(signature_version=botocore.UNSIGNED)
        )


thread_pool = None


def set_thread_pool():
    global thread_pool
    if not thread_pool:
        thread_pool = ThreadPoolExecutor(initializer=set_global_session)


def get_warc(filename, offset, length, existing_compressed_warc):
    if existing_compressed_warc is not None:
        return existing_compressed_warc, None

    try:
        response = thread_data.s3_client.get_object(
            Bucket="commoncrawl",
            Key=filename,
            Range=f"bytes={offset}-{offset + length - 1}",
        )
    except (ClientError, botocore.exceptions.ProxyConnectionError) as e:
        return None, repr(e)

    # Check error handling
    return response["Body"].read(), None


def get_warcs(batch):
    """We compose both as `get_outgoing_links` checks the WARC quality"""
    warc_filenames = batch["warc_filename"]
    warc_record_length = batch["warc_record_length"]
    warc_record_offset = batch["warc_record_offset"]
    assert len(warc_filenames) == len(warc_record_length)
    assert len(warc_filenames) == len(warc_record_offset)

    if "compressed_warc" in batch:
        existing_compressed_warcs = batch["compressed_warc"]
    else:
        # Not yet queried
        existing_compressed_warcs = [None] * len(warc_filenames)

    set_thread_pool()
    global thread_pool
    warcs_or_exceptions = thread_pool.map(
        get_warc,
        warc_filenames,
        warc_record_offset,
        warc_record_length,
        existing_compressed_warcs,
    )

    batch["compressed_warc"], batch["download_exception"] = [
        list(l) for l in zip(*warcs_or_exceptions)
    ]
    return batch


def download_warcs(ds, save_path, num_proc):
    # Get raw compressed WARC records
    ds = ds.map(
        get_warcs,
        batched=True,
        num_proc=num_proc,
        features=datasets.Features(
            {
                **ds.features,
                "compressed_warc": datasets.Value("binary"),
                "download_exception": datasets.Value("string"),
            }
        ),
    )

    # Provide a way to re-run the script where we query only the files that failed download.
    logger.info(
        f"Download failed for {len([e for e in ds['download_exception'] if e is not None])} rows. Please try re-running this script somehow."
    )

    ds.save_to_disk(f"{str(save_path.absolute())}.tmp")
    subprocess.run(
        ["mv", f"{str(save_path.absolute())}.tmp", str(save_path.absolute())]
    )

    with open(save_path / "missing_rows.txt", "w") as fi:
        indices_that_failed = [
            i for i, e in enumerate(ds["download_exception"]) if e is not None
        ]
        fi.write(
            f"Download failed for {len(indices_that_failed)} rows. Please try re-running this script somehow.\n"
        )
        fi.writelines([f"{i}\n" for i in indices_that_failed])


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

    if not args.use_datasets_caching:
        datasets.set_caching_enabled(False)
    else:
        logger.info(
            f"the datasets results will be cached at {config.HF_DATASETS_CACHE}."
        )

    if args.shard_id is not None:
        save_path = (
            Path(args.save_dir) / f"{args.dataset}--{args.shard_id}--{args.num_shards}"
        )
    else:
        save_path = Path(args.save_dir) / args.dataset

    if save_path.exists():
        print(f"Folder {save_path.absolute()} already exists.")
        return

    ds = load_dataset(
        "parquet",
        data_files=[
            f"{args.cc_index_folder}/subset=warc/*",
            f"{args.cc_index_folder}/**/subset=warc/*",
        ],
        split=f"train{f'[{args.range}]' if args.range is not None else ''}",
    )

    if args.shard_id is not None:
        ds = ds.shard(num_shards=args.num_shards, index=args.shard_id)

    download_warcs(ds, save_path, args.num_proc)


if __name__ == "__main__":
    main()
