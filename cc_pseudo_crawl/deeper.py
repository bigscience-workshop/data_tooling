import csv
import os
import subprocess
from argparse import ArgumentParser
from pathlib import Path
import urllib.parse

from datasets import load_dataset

from .preprocess_dataset import get_all_parquet_files


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name")

    args = parser.parse_args()

    return args

def intermediate_next(url_candidates, previous_urls):
    """Query only those urls"""
    new_urls_to_query = set(url_candidates) - set(previous_urls)
    return new_urls_to_query

# def extended_next(url_candidates, previous_urls):
#     """Query new domains"""
#     def get_domain(url):
#         parsed_url = urllib.parse.urlparse(url)
#         return parsed_url.netloc
#     new_domains_to_query = set(get_domain(url) for url in url_candidates) - set(get_domain(url) for url in previous_urls)
#     return new_domains_to_query

def main():
    args = get_args()
    csv_output_dir = Path(__file__).parent / "temp"
    subprocess.run(["mkdir", "-p", str(csv_output_dir.absolute())])

    # Load previous depth dataset
    previous_ds = load_dataset(args.dataset, use_auth_token=True)

    previous_depth = max(previous_ds["depth"])
    url_candidates = set([url for external_urls in previous_ds["external_urls"] for url in external_urls])
    previous_urls = set(previous_ds["url"])

    intermediate_depth_urls = intermediate_next(url_candidates, previous_urls)
    # extended_depth_domains = extended_next(url_candidates, previous_urls)

    new_depth = previous_depth + 1
    id_offset = max(previous_depth["id"]) + 1

    with open(csv_output_dir / f"intermediate_depth_{new_depth}.csv", "w") as fo:
        writer = csv.writer(fo)
        writer.writerow(["id", "url", "depth"])
        for i, url in enumerate(intermediate_depth_urls):
            writer.writerow([id_offset + i, url, new_depth])

    # with open(csv_output_dir / f"extended_depth_{new_depth}.csv", "w") as fo:
    #     writer = csv.writer(fo)
    #     writer.writerow(["id", "domain", "depth"])
    #     for i, domain in enumerate(extended_depth_domains):
    #         writer.writerow([id_offset + i, domain, new_depth])

    # For each url, find the most recent row id corresponding to that url
    # All of the duplicate of a `url` are either all in that dictionary or not in that dictionary
    # This table allows me to do a double join so I can easily compute the ids.
    # We'll then go through that csv and add the ids to final dataset.
    # No duplicates guaranteed
    url_to_id = {}
    with open(csv_output_dir / f"previous_to_next.csv", "w") as fo:
        writer = csv.writer(fo)
        writer.writerow(['previous_id', 'previous_url', 'next_id', 'next_url'])
        for data in previous_ds:
            previous_id = data["id"]
            previous_url = data["url"]
            for external_url in data["external_urls"]:
                next_id = url_to_id.get(external_url, None)
                writer.writerow([previous_id, previous_url, next_id, external_url])


if __name__ == "__main__":
    main()
