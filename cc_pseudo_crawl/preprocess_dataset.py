import functools
import io
import re
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path
from queue import Queue

import boto3
import botocore
import datasets
from bs4 import BeautifulSoup
from datasets import load_dataset
# DEBUG
datasets.set_caching_enabled(False)
from botocore.config import Config

from warcio.archiveiterator import WARCIterator
from warcio.recordloader import ArchiveLoadFailed

"""
Required: obtain cc_index and copy it locally
`aws s3 sync s3://commoncrawl-dev/big-science-workshop/data-sourcing-sheet/cc/ $CC_INDEX_FOLDER/cc/`
"""

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name")
    parser.add_argument('--cc-index-folder', type=str, required=True, help="Folder containing index dataset in parquet format")
    parser.add_argument('--save-dir', type=str, required=True, help="Where to save the datasets.")
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
        s3_client = boto3.session.Session().client('s3', config=Config(signature_version=botocore.UNSIGNED))

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

def get_warc(filename, offset, length):
    global s3_client
    response = s3_client.get_object(
        Bucket='commoncrawl',
        Key=filename,
        Range=f"bytes={offset}-{offset + length - 1}"
    )
    # Check error handling
    return response["Body"].read()

def get_outgoing_link(compressed_warc, mime, domain):
    if mime not in HTML_TYPES:
        return None

    with io.BytesIO(compressed_warc) as stream:
        html = None
        try:
            for record in WARCIterator(stream):
                if record.rec_type == 'response':
                    html = record.content_stream().read()
                    break
        except ArchiveLoadFailed as exception:
            print(str(exception), compressed_warc)
            raise exception

    assert html is not None
    soup = BeautifulSoup(html, 'html.parser')
    return get_external_links(soup, domain)

def add_to_list_when_consuming(generator, list_to_cumulate):
    for elt in generator:
        list_to_cumulate.append(elt)
        yield elt

HTML_TYPES = ['text/html', 'application/xhtml+xml']
def get_warc_and_outgoing_links(batch, thread_pool):
    """We compose both as `get_outgoing_links` checks the WARC quality"""
    content_mime_detected = batch["content_mime_detected"]  # select only text/html
    # url_host_registered_domains = batch["url_host_registered_domain"]
    warc_filenames = batch["warc_filename"]
    warc_record_length = batch["warc_record_length"]
    warc_record_offset = batch["warc_record_offset"]
    assert len(content_mime_detected) == len(warc_filenames)
    assert len(content_mime_detected) == len(warc_record_length)
    assert len(content_mime_detected) == len(warc_record_offset)

    # TODO: Try using ThreadPoolExecutor download the files in a threadpool

    warcs = thread_pool.map(get_warc, warc_filenames, warc_record_offset, warc_record_length)
    compressed_warcs = list(warcs)

    # compressed_warcs = []
    # warc_generator = add_to_list_when_consuming(warcs, compressed_warcs)
    # external_urls = process_pool.starmap(get_outgoing_link, zip(warc_generator, content_mime_detected , url_host_registered_domains))

    batch["compressed_warc"] = compressed_warcs
    # batch["external_urls"] = external_urls
    return batch

def assign_depth(batch, depth):
    batch_size = len(batch[next(iter(batch))])
    batch["depth"] = [depth]* batch_size
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

    if args.shard_id is not None:
        ds = ds.shard(num_shards=args.num_shards, index=args.shard_id)

    # Get raw compressed WARC records and outgoing links
    with ThreadPoolExecutor(5 * args.num_proc, initializer=set_global_session) as thread_pool:
        ds = ds.map(
            functools.partial(get_warc_and_outgoing_links, thread_pool = thread_pool),
            batched=True,
            num_proc=1 # multiprocessing is handled manually
        )

    # # Assign depth.
    # ds = ds.map(functools.partial(assign_depth, depth=get_depth(args.flavor)), batched=True, num_proc=args.num_proc)

    if args.shard_id:
        save_path = Path(args.save_dir) / f"{args.dataset}--{args.shard_id}--{args.num_shards}"
    else:
        save_path = Path(args.save_dir) / args.dataset
    ds.save_to_disk(save_path)
    # # Clean up columns to keep only these ones
    # columns_to_keep = {"id", "seed_id", "title", "link", "languages", "url", "pdf_url", "compressed_warc",
    #                    "external_urls", "depth", "fetch_time"}
    # columns_to_remove = [column for column in ds.column_names if column not in columns_to_keep]
    # ds = ds.remove_columns(columns_to_remove)

    # ds.push_to_hub(args.dataset, private=True)

if __name__ == "__main__":
    main()
