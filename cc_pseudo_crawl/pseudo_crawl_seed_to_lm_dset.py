import argparse
import gzip
import json
import os

import datasets
from datasets import Features, load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

###
# features of the pseudocrawl seeds
###

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
    meta = dict([(k, page[k]) for k in ["url", "content_languages", "seed_id"]])
    return meta


# filter text to remove certain lines (e.g. menu items, copyright notice)
def filter_lines(article, skip_dict):
    lines = [line.strip() for line in article.split("\n")]
    keep = []
    skip = []
    for i, line in enumerate(lines):
        if skip_dict.get(line, False):
            skip += [line]
        else:
            keep += [line]
    return "\n".join(keep), "\n".join(skip)


# do both together and return an entry
def process_page(page, skip_dict):
    meta = get_meta_dict(page)
    text, _ = filter_lines(page["text"], skip_dict)
    return {
        "meta": meta,
        "text": text,
    }


# looks at up to the first 10K pages for a seed and
# records lines that appear in at least 1% of the unique pages
def get_lines_to_skip(dset):
    line_counts = {}
    seen_pages = {}
    dset_sample = dset["train"].select(range(10000))
    for page in tqdm(dset_sample):
        article = page["text"]
        if not seen_pages.get(article, False):
            seen_pages[article] = True
            for line in article.split("\n"):
                line_counts[line.strip()] = line_counts.get(line.strip(), 0) + 1
    thres_skip = max(10, len(seen_pages) // 100)
    skip_dict = dict(
        [(line, True) for line, ct in line_counts.items() if ct > thres_skip]
    )
    return skip_dict


# create a private repository and push processed seed in jsonl format
def push_seed_to_hub(
    dset, language, name, token, skip_lines_dict, min_chars=32, gzipped=False
):
    repo_name = f"lm_{language}_pseudocrawl_{name}"
    # process and write to file
    if gzipped:
        file_name = f"{repo_name}.jsonl.gz"
        fout = gzip.open("lm_vi_pseudocrawl_vnexpress.jsonl.gz", "w")
        for article in tqdm(dset["train"]):
            processed_dct = process_page(article, skip_lines_dict)
            if len(processed_dct["text"]) > min_chars:
                _ = fout.write((json.dumps(processed_dct) + "\n").encode("utf-8"))
        fout.close()
    else:
        file_name = f"{repo_name}.jsonl"
        f = open(file_name, "w", encoding="utf-8")
        for article in tqdm(dset["train"]):
            processed_dct = process_page(article, skip_lines_dict)
            if len(processed_dct["text"]) > min_chars:
                _ = f.write(json.dumps(processed_dct) + "\n")
        f.close()
    # upload to hub
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
        token=_AUTH_TOKEN,
        repo_type="dataset",
    )
    return file_loc


###
# combine everything
###
def main():
    parser = argparse.ArgumentParser(description="Load seed and upload to hub")
    parser.add_argument(
        "-sid",
        "--seed_id",
        help="seed ID",
        required=True,
        type=int,
    )
    parser.add_argument(
        "-ln",
        "--language_code",
        help="language code used on the repo",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-n",
        "--name",
        help="name of the website",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-t",
        "--token",
        help="authentication token with write access to the org",
        required=True,
        type=str,
    )
    args = parser.parse_args()
    # Load dataset (data first needs to be git pulled, see above)
    dset = load_dataset(
        "json",
        data_files=[
            f"{_PATH_TO_PSEUDO_CRAWL}/seed_id={args.seed_id}/text__html/*.jsonl.gz"
        ],
        features=final_features,
        cache_dir=f"cache_seed_{args.seed_id}",
    )
    skip_lines_dict = get_lines_to_skip(dset)
    push_seed_to_hub(
        dset,
        language=args.language_code,
        name=args.name,
        token=args.token,
        skip_lines_dict=skip_lines_dict,
        min_chars=32,
        gzipped=False,  # TODO: switch to using a gzipped version once the hub regex validation is fixed
    )


if __name__ == "__main__":
    main()
