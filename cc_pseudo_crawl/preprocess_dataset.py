import io
import re
from argparse import ArgumentParser
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from datasets import load_dataset
from warcio import ArchiveIterator

"""
Required: obtain cc_index and copy it locally
`aws s3 sync s3://commoncrawl-dev/big-science-workshop/data-sourcing-sheet/cc/ $CC_INDEX_FOLDER/cc/`
"""

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--cc-index-folder', type=str, required=True, help="Folder containing index dataset in parquet format")
    parser.add_argument('--num-proc', type=int, default=1, help="Number of procs use for preprocessing")
    parser.add_argument('--range', type=str, default=None, help="Optional argument to select a subset (used for debugging purposes). Example `:10`")
    args = parser.parse_args()

    args.cc_index_folder = Path(args.cc_index_folder) / "cc"

    return args

def get_all_parquet_files(path):
    path = Path(path)
    all_crawls = [crawl for crawl in path.iterdir() if crawl.is_dir()]
    only_warcs = [subset for crawl in all_crawls for subset in crawl.iterdir() if subset.is_dir() and subset.name == "subset=warc"]
    return [str(file.absolute().resolve()) for subset in only_warcs for file in subset.iterdir() if file.is_file()]

def get_pdf_urls(batch):
    content_mime_detected = batch["content_mime_detected"]
    urls = batch["url"]
    assert len(content_mime_detected) == len(urls)
    # Arrow doesn't support None, setting empty string for now
    batch["pdf_url"] = [url if mime == "application/pdf" else None for mime, url in zip(content_mime_detected, urls)]
    return batch

HTML_TYPES = ['text/html', 'application/xhtml+xml']
def get_warc(batch):
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
        headers = {
            "Range": f"bytes={offset}-{offset + length - 1}"
        }

        response = requests.get(f'https://commoncrawl.s3.amazonaws.com/{filename}', headers=headers)
        compressed_warcs.append(response.content)

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
            for record in ArchiveIterator(stream):
                if record.rec_type == 'response':
                    html = record.content_stream().read()
                    break

        assert html is not None
        soup = BeautifulSoup(html, 'html.parser')
        external_urls.append(get_external_links(soup, domain))

    batch["external_urls"] = external_urls
    return batch

def main():
    args = get_args()

    ds = load_dataset("parquet", data_files=get_all_parquet_files(args.cc_index_folder), split=f"train{f'[{args.range}]' if args.range is not None else ''}")

    ds = ds.map(get_warc, batched=True, num_proc=args.num_proc)

    ds = ds.map(get_outgoing_links, batched=True, num_proc=args.num_proc)

    columns_to_keep = ["id", "title", "link", "languages", "pdf_url", "html", "compressed_warc", "external_urls"]
    columns_to_remove = [column for column in ds.column_names if column not in columns_to_keep]
    cleaned_ds = ds.remove_columns(columns_to_remove)

    cleaned_ds.push_to_hub("bigscience-catalogue-data/pseudo_crawl_test", private=True)


if __name__ == "__main__":
    main()