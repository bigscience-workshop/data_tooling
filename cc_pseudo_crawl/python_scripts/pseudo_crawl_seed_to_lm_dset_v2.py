from collections import defaultdict
import os
import argparse
import logging
import gzip
import json

import datasets
from functools import partial
import pandas as pd
from datasets import Features, load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm
from datasets.utils.logging import set_verbosity_info

###
# features of the pseudocrawl seeds
###

set_verbosity_info()
logger = logging.getLogger(__name__)

_PATH_TO_PSEUDO_CRAWL = "pseudo_crawl"

null = None
features = {
    "HtmlPreprocessor_error": {"dtype": "int64", "id": null, "_type": "Value"},
    "HtmlPreprocessor_error_comment": {"dtype": "string", "id": null, "_type": "Value"},
    "content_languages": {"dtype": "string", "id": null, "_type": "Value"},
    "content_mime_detected": {"dtype": "string", "id": null, "_type": "Value"},
    "depth": {"dtype": "int16", "id": null, "_type": "Value"},
    "download_exception": {"dtype": "string", "id": null, "_type": "Value"},
    "external_urls": [{"dtype": "string", "id": null, "_type": "Value"}],
    "fetch_redirect": {"dtype": "string", "id": null, "_type": "Value"},
    "fetch_status": {"dtype": "int32", "id": null, "_type": "Value"},
    "fetch_time": {"dtype": "timestamp[ns]", "id": null, "_type": "Value"},
    "html_error": {"dtype": "string", "id": null, "_type": "Value"},
    "html_footer": [{"dtype": "string", "id": null, "_type": "Value"}],
    "html_head": [{"dtype": "string", "id": null, "_type": "Value"}],
    "html_str": {"dtype": "string", "id": null, "_type": "Value"},
    "html_title": [{"dtype": "string", "id": null, "_type": "Value"}],
    "metadata_html": [
        {
            "char_end_idx": {"dtype": "int64", "id": null, "_type": "Value"},
            "char_start_idx": {"dtype": "int64", "id": null, "_type": "Value"},
            "html_attrs": {
                "attrs": [{"dtype": "string", "id": null, "_type": "Value"}],
                "values": [{"dtype": "string", "id": null, "_type": "Value"}],
            },
            "key": {"dtype": "string", "id": null, "_type": "Value"},
            "relative_end_pos": {"dtype": "int64", "id": null, "_type": "Value"},
            "relative_start_pos": {"dtype": "int64", "id": null, "_type": "Value"},
            "type": {"dtype": "string", "id": null, "_type": "Value"},
            "value": {"dtype": "string", "id": null, "_type": "Value"},
        }
    ],
    "seed_id": {"dtype": "int32", "id": null, "_type": "Value"},
    "text": {"dtype": "string", "id": null, "_type": "Value"},
    "url": {"dtype": "string", "id": null, "_type": "Value"},
    "url_host_name": {"dtype": "string", "id": null, "_type": "Value"},
    "url_host_registered_domain": {"dtype": "string", "id": null, "_type": "Value"},
    "url_host_tld": {"dtype": "string", "id": null, "_type": "Value"},
    "url_surtkey": {"dtype": "string", "id": null, "_type": "Value"},
    "warc_filename": {"dtype": "string", "id": null, "_type": "Value"},
    "warc_record_length": {"dtype": "int32", "id": null, "_type": "Value"},
    "warc_record_offset": {"dtype": "int32", "id": null, "_type": "Value"},
}


def convert_types(features):
    if isinstance(features, dict) and "_type" in features:
        return getattr(datasets, features["_type"])(features["dtype"])
    elif isinstance(features, dict):
        return {key: convert_types(value) for key, value in features.items()}
    elif isinstance(features, list):
        return [convert_types(value) for value in features]


final_features = convert_types(features)
final_features = Features(final_features)
final_features

###
# seed processing and upload functions
###


# extract just the metadata we wish to keep
def get_meta_dict(page):
    meta = {k: page[k] for k in ["url", "content_languages", "seed_id"]}
    return meta


# filter text to remove certain lines (e.g. menu items, copyright notice)
def filter_lines(article, skip_set):
    # TODO discuss the strip
    lines = [line.strip() for line in article.split("\n")]
    keep = []
    skip = []
    for line in lines:
        if line in skip_set:
            skip += [line]
        else:
            keep += [line]
    return "\n".join(keep).strip(), "\n".join(skip).strip()


# do both together and return an entry
def process_page(page, skip_set):
    meta = get_meta_dict(page)
    text, _ = filter_lines(page["text"], skip_set)
    return {
        "meta": meta,
        "text": text,
    }


# looks at up to the first 10K pages for a seed and
# records lines that appear in at least 1% of the unique pages
def get_lines_to_skip(dset, n_records=10000, pourcentage_threshold=0.01):
    line_counts = defaultdict(lambda: 0)
    seen_pages = set()
    dset_sample = dset["train"].select(range(n_records))
    for page in tqdm(dset_sample):
        article = page["text"]
        if article not in seen_pages:
            seen_pages.add(article)
            for line in article.split("\n"):
                line_counts[line.strip()] += 1
    # TODO understand this logic, why it's not len(line_counts)
    thres_skip = max(10, len(seen_pages) * pourcentage_threshold)
    skip_set = set(line for line, ct in line_counts.items() if ct > thres_skip)
    return skip_set

def clean_examples(examples, skip_lines_set, args):
    results = {
        "text": [],
        "meta": []
    }
    for idx, article in enumerate(examples["text"]):
        examples["text"][idx] = process_page(article, skip_lines_set)
        if len(examples["text"][idx]) <= args.min_chars:
            continue
        for key in results.keys():
            results[key].append(examples[key][idx])

    return results

# create a private repository and push processed seed in jsonl format
def make_seed_jsonl(dset, skip_lines_set, args): # language, name, seed_id, num_proc, min_chars=32, gzipped=False, save_dir=None):
    dset = dset.map(
        partial(clean_examples, skip_lines_set=skip_lines_set, args=args),
        batched=True,
        num_proc=args.num_proc,
        batch_size=args.batch_size
    )

    repo_name = f"lm_{args.language}_seed_id_{args.seed_id}_pseudocrawl_{args.name}"
    if args.save_dir is not None:
        repo_name = os.path.join(args.save_dir, repo_name)
    file_name = os.path.join(repo_name, f"data.jsonl.gz")
    logger.info(f"the dataset will be saved at {file_name}")
    if not os.path.isdir(repo_name):
        os.makedirs(path)
    if args.gzipped:
        dset.to_json(
            repo_name,
            num_proc=args.num_proc,
            batch_size=args.save_batch_size,
            compression="gzip",
        )
    else:
        dset.to_json(
            repo_name,
            num_proc=args.num_proc,
            batch_size=args.save_batch_size,
        )




    # process and write to file
        
        f = gzip.open(file_name, "w")
    else:
        file_name = f"{repo_name}.jsonl"
        f = open(file_name, "w", encoding="utf-8")
    duplicated = {}
    for article in tqdm(dset["train"]):
        processed_dct = process_page(article, skip_lines_dict)
        txt = processed_dct["text"].strip().lower()
        if len(processed_dct["text"]) > min_chars and txt not in duplicated:
            _ = f.write((json.dumps(processed_dct) + "\n").encode("utf-8").decode("utf-8"))
    f.close()
    return file_name, repo_name


def push_jsonl_to_hub(file_name, repo_name, token):
    api = HfApi()
    api.create_repo(
        f"bigscience-catalogue-lm-data/{repo_name}",
        private=True,
        repo_type="dataset",
        token=token,
    )
    file_loc = api.upload_file(
        path_or_fileobj=file_name,
        path_in_repo=file_name,
        repo_id=f"bigscience-catalogue-lm-data/{repo_name}",
        token=token,
        repo_type="dataset",
    )
    return file_loc


def get_dataset_name_and_lang_id_from_seed_id(seed_id, seed_id_info_path):
    df = pd.read_csv(seed_id_info_path)
    sub_df = df[df["id"] == seed_id]
    if len(sub_df) != 1:
        raise ValueError("You should have only one match per seed id")
    name = sub_df.name[0]
    lang_id = sub_df.lang_id[0]
    return name, lang_id


def get_dataset_name_and_lang_id_from_seed_id_fake(seed_id, seed_id_info_path):
    return "change_name", "change_lang_id"


###
# combine everything
###
def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(description="Load seed and upload to hub")
    parser.add_argument(
        "-sid",
        "--seed-id",
        help="seed ID",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--seed-id-info-path",
        help="The path to a csv containing the seed id and the corresponding lang-id and name",
        # required=True,
        type=str,
    )
    parser.add_argument("--save-dir", required=True, type=str, help="Where to save the datasets.")
    parser.add_argument(
        "-pc_path",
        "--pseudo_crawl_path",
        help="path to where the pseudocrawl is located",
        default="pseudo_crawl",
        type=str,
    )
    parser.add_argument(
        "-gz",
        "--gzipped",
        help="Write file directly in jsonl.gz compressed format",
        action="store_true",
    )
    parser.add_argument(
        "-hub",
        "--push_to_hub",
        help="Whether to create a repository and push the computed jsonl to the hub",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--token",
        help="authentication token with write access to the org",
        default="",
        type=str,
    )
    args = parser.parse_args()
    assert not (
        args.push_to_hub and args.token == ""
    ), "If you want the script to push to the hub, you need to provide an authentication token"
    # Load dataset (data first needs to be git pulled, see above)
    dset = load_dataset(
        "json",
        data_files=[f"{args.pseudo_crawl_path}/seed_id={args.seed_id}/text__html/*.jsonl.gz"],
        features=final_features,
    )

    name, language_code = get_dataset_name_and_lang_id_from_seed_id_fake(args.seed_id, args.seed_id_info_path)
    skip_lines_dict = get_lines_to_skip(dset)
    file_name, repo_name = make_seed_jsonl(
        dset,
        language=language_code,
        name=name,
        seed_id=args.seed_id,
        skip_lines_dict=skip_lines_dict,
        min_chars=128,  # only keep examples with at least 128 characters
        gzipped=args.gzipped,
        save_dir=args.save_dir,
    )
    if args.push_to_hub:
        push_jsonl_to_hub(file_name, repo_name, args.token)


if __name__ == "__main__":
    main()
