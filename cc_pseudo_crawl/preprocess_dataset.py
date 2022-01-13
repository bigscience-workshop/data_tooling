import functools
import io
import re
from argparse import ArgumentParser
from pathlib import Path
from queue import Queue

import boto3
import botocore
import datasets
from bs4 import BeautifulSoup
from datasets import load_dataset
from botocore.config import Config

from warcio import ArchiveIterator
from warcio.recordloader import ArchiveLoadFailed

"""
Required: obtain cc_index and copy it locally
`aws s3 sync s3://commoncrawl-dev/big-science-workshop/data-sourcing-sheet/cc/ $CC_INDEX_FOLDER/cc/`
"""

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name")
    parser.add_argument('--cc-index-folder', type=str, required=True, help="Folder containing index dataset in parquet format")
    parser.add_argument('--num-proc', type=int, default=1, help="Number of procs use for preprocessing")
    parser.add_argument('--range', type=str, default=None, help="Optional argument to select a subset (used for debugging purposes). Example `:10`")
    parser.add_argument('--shard-id', type=int, help="Preprocess dataset via shards")
    parser.add_argument('--num-shards', type=int, help="Total number of shards")

    args = parser.parse_args()

    matches = re.match(r"^bigscience-catalogue-data/pseudo_crawl_(?:(.*)_partial|(seed))$", args.dataset)
    assert matches is not None
    flavors = [elt for elt in matches.groups() if elt is not None]
    assert len(flavors) == 1
    flavor = flavors[0]
    assert flavor == "seed" \
           or re.match(r"^intermediate_depth_([0-9]+)$", flavor) is not None
    args.cc_index_folder = Path(args.cc_index_folder) / f"cc-{flavor}"
    args.flavor = flavor

    if args.shard_id is not None:
        assert args.num_shards is not None

    return args

def get_all_parquet_files(path):
    path = Path(path)

    def add_parquet_files(path):
        return [str(file.absolute().resolve()) for file in path.iterdir() if file.is_file()]

    parquet_files = []
    queue_dirs = Queue()
    queue_dirs.put(path)
    while not queue_dirs.empty():
        dir_path = queue_dirs.get()
        if dir_path.name == "subset=warc":
            parquet_files += add_parquet_files(dir_path)
        for d in dir_path.iterdir():
            if d.is_dir():
                queue_dirs.put(d)

    return parquet_files

def get_pdf_urls(batch):
    content_mime_detected = batch["content_mime_detected"]
    urls = batch["url"]
    assert len(content_mime_detected) == len(urls)
    # Arrow doesn't support None, setting empty string for now
    batch["pdf_url"] = [url if mime == "application/pdf" else None for mime, url in zip(content_mime_detected, urls)]
    return batch

s3_client = None
def set_global_session():
    global s3_client
    if not s3_client:
        s3_client = boto3.client('s3', config=Config(signature_version=botocore.UNSIGNED))

def get_warc(batch):
    set_global_session()
    global s3_client
    content_mime_detected = batch["content_mime_detected"]  # select only text/html
    url_host_registered_domains = batch["url_host_registered_domain"]
    warc_filenames = batch["warc_filename"]
    warc_record_length = batch["warc_record_length"]
    warc_record_offset = batch["warc_record_offset"]
    assert len(content_mime_detected) == len(warc_filenames)
    assert len(content_mime_detected) == len(warc_record_length)
    assert len(content_mime_detected) == len(warc_record_offset)

    compressed_warcs = []
    for mime, filename, length, offset, domain in zip(content_mime_detected, warc_filenames, warc_record_length,
                                                      warc_record_offset, url_host_registered_domains):
        response = s3_client.get_object(
            Bucket='commoncrawl',
            Key=filename,
            Range=f"bytes={offset}-{offset + length - 1}"
        )
        content = response["Body"].read()
        compressed_warcs.append(content)

    batch["compressed_warc"] = compressed_warcs
    return batch

# Retrieves a list of all external links found on a page
def get_external_links(soup, exclude_url):
    external_links = set()
    # Finds all links that start with "http" that do
    # not contain the current URL
    for link in soup.find_all('a', {'href': re.compile('^(((http|https)://)|www){1,2}((?!' + exclude_url + ').)*$')}):
        href = link.attrs['href']
        if href is not None:
            external_links.add(href)
    return list(external_links)

HTML_TYPES = ['text/html', 'application/xhtml+xml']
def get_outgoing_links(batch):
    content_mime_detected = batch["content_mime_detected"]  # select only text/html
    url_host_registered_domains = batch["url_host_registered_domain"]
    compressed_warcs = batch["compressed_warc"]
    assert len(content_mime_detected) == len(compressed_warcs)
    assert len(content_mime_detected) == len(url_host_registered_domains)

    external_urls = []
    for mime, compressed_warc, domain in zip(content_mime_detected, compressed_warcs, url_host_registered_domains):
        if mime not in HTML_TYPES:
            external_urls.append([])
            continue

        with io.BytesIO(compressed_warc) as stream:
            html = None
            try:
                for record in ArchiveIterator(stream):
                    if record.rec_type == 'response':
                        html = record.content_stream().read()
                        break
            except ArchiveLoadFailed as exception:
                print(str(exception), compressed_warc)
                raise exception

        assert html is not None
        soup = BeautifulSoup(html, 'html.parser')
        external_urls.append(get_external_links(soup, domain))

    batch["external_urls"] = external_urls
    return batch

def get_warc_and_outgoing_links(batch):
    """We compose both as `get_outgoing_links` checks the WARC quality"""
    batch = get_warc(batch)
    return get_outgoing_links(batch)

def assign_depth(batch, depth):
    batch["depth"] = [depth]*len(batch["id"])
    return batch

def get_depth(flavor):
    if flavor == "seed":
        return 0
    else:
        # TODO: fix for extended_depth
        empty, depth = flavor.split("intermediate_depth_")
        assert empty == ""
        return int(depth)

def main():
    args = get_args()

    ds = load_dataset("parquet", data_files=get_all_parquet_files(args.cc_index_folder), split=f"train{f'[{args.range}]' if args.range is not None else ''}")

    if args.shard_id:
        ds = ds.shard(num_shards=args.num_shards, index=args.shard_id)

    # Get raw compressed WARC records and outgoing links
    ds = ds.map(
        get_warc_and_outgoing_links,
        batched=True,
        num_proc=args.num_proc
    )

    datasets.set_caching_enabled(False)
    # Assign depth.
    ds = ds.map(functools.partial(assign_depth, depth=get_depth(args.flavor)), batched=True, num_proc=args.num_proc)

    # # Clean up columns to keep only these ones
    # columns_to_keep = {"id", "seed_id", "title", "link", "languages", "url", "pdf_url", "compressed_warc",
    #                    "external_urls", "depth", "fetch_time"}
    # columns_to_remove = [column for column in ds.column_names if column not in columns_to_keep]
    # ds = ds.remove_columns(columns_to_remove)

    # ds.push_to_hub(args.dataset, private=True)

if __name__ == "__main__":
    main()
