from asyncio import subprocess
import functools
import io
import logging
from pathlib import Path
import re
from argparse import ArgumentParser

from bs4 import BeautifulSoup
import datasets
from datasets import config, load_from_disk
from datasets.utils.logging import set_verbosity_info
from warcio.archiveiterator import WARCIterator
from warcio.exceptions import ArchiveLoadFailed


set_verbosity_info()
logger = logging.getLogger(__name__)

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True, help="path to the parquet dataset folder")
    parser.add_argument('--save-path', type=str, help="Where to save the datasets.")
    parser.add_argument('--use-datasets-caching', action='store_true')
    parser.add_argument('--num-proc', type=int, default=1, help="Number of procs use for preprocessing.")
    parser.add_argument('--flavor', type=str, required=True, default=1, help="Number of procs use for preprocessing.")

    args = parser.parse_args()

    return args

def get_pdf_urls(batch):
    content_mime_detected = batch["content_mime_detected"]
    urls = batch["url"]
    assert len(content_mime_detected) == len(urls)
    # Arrow doesn't support None, setting empty string for now
    batch["pdf_url"] = [url if mime == "application/pdf" else None for mime, url in zip(content_mime_detected, urls)]
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

def get_beautifulsoup_object(compressed_warc, mime):
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
    return soup

def get_html_str_and_outgoing_link(compressed_warc, mime, domain):
    soup = get_beautifulsoup_object(compressed_warc, mime)
    if soup is None:
        return None, None

    html_str = soup.decode_contents(formatter="html")
    external_links = get_external_links(soup, domain)
    return html_str, external_links


def assign_depth_(batch, depth):
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

def apply_preprocessing(batch, depth):
    """Wrapper class to hold all transforms into a single function."""
    content_mime_detected = batch["content_mime_detected"]  # select only text/html
    url_host_registered_domains = batch["url_host_registered_domain"]
    compressed_warcs = batch["compressed_warc"]

    assert len(content_mime_detected) == len(url_host_registered_domains)
    assert len(content_mime_detected) == len(compressed_warcs)

    batch["external_urls"] = []
    batch["html_str"] = []
    for compressed_warc, mime, domain in zip(content_mime_detected, url_host_registered_domains, compressed_warcs):
        html_str, external_links = get_html_str_and_outgoing_link(compressed_warc, mime, domain)
        batch["external_urls"].append(external_links)
        batch["html_str"].append(html_str)
    
    assign_depth_(batch, depth)

    return batch

def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = get_args()
    logger.info(f"** The job is runned with the following arguments: **\n{args}\n **** ")

    if not args.use_datasets_caching:
        datasets.set_caching_enabled(False)
    else:
        logger.info(f"the datasets results will be cached at {config.HF_DATASETS_CACHE}.")
    
    ds = load_from_disk(args.dataset_path)

    ds.map(
            functools.partial(apply_preprocessing, depth=args.flavor),
            batched=True,
            num_proc=args.num_proc
        )
        
    if args.save_path:
        save_path = Path(args.save_path)
    else:
        save_path = Path(args.dataset_path)

    save_path_tmp = f"{str(save_path.absolute())}.tmp"
    logger.info(f"Saving the dataset at {save_path_tmp}")
    ds.save_to_disk(save_path_tmp)
    logger.info(f"Moving the saved dataset to {str(save_path.absolute())}")
    subprocess.run(["mv", save_path_tmp, str(save_path.absolute())])


if __name__ == "__main__":
    main()
