import functools
import io
import re
from argparse import ArgumentParser

from bs4 import BeautifulSoup
from datasets import load_dataset, concatenate_datasets
from warcio.archiveiterator import WARCIterator
from warcio.exceptions import ArchiveLoadFailed


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset-prefix-path', type=str, required=True, help="Dataset name. Essentially we're going to search for all the shards")
    parser.add_argument('--num-shards', type=int, help="Number of shards we need to query.")
    parser.add_argument('--num-proc', type=int, default=1, help="Number of procs use for preprocessing.")

    args = parser.parse_args()

    matches = re.match(r"^bigscience-catalogue-data/pseudo_crawl_(?:(.*)_partial|(seed))(_dedup_url)?$", args.dataset)
    assert matches is not None
    flavors = [elt for elt in matches.groups() if elt is not None]
    assert len(flavors) == 1 or (len(flavors) == 2 and flavors[1] == "_dedup_url")
    flavor = flavors[0]
    assert flavor == "seed" \
           or re.match(r"^intermediate_depth_([0-9]+)$", flavor) is not None
    args.flavor = flavor

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

    batch["external_urls"] = [
        get_outgoing_link(compressed_warcs, mime, domain)
        for compressed_warc, mime, domain in zip(content_mime_detected, url_host_registered_domains, compressed_warcs)
    ]
    assign_depth_(batch, depth)

    return batch

def concatenate_all_shards(args):
    if args.num_shards is None:
        source_path = args.dataset_prefix_path
        ds = load_dataset(source_path)
    else:
        shard_paths = [f"{args.dataset_prefix_path}--{i}--{args.num_shards}" for i in range(args.num_shards)]
        shards = [load_dataset(shard_path) for shard_path in shard_paths]
        ds = concatenate_datasets(shards)
    return ds

def main():
    args = get_args()

    # Actually keeps shards, and concatenate after preprocessing.
    ds = concatenate_all_shards(args)

    ds.map(
        functools.partial(apply_preprocessing, depth=args.flavor),
        batched=True,
        num_proc=args.num_proc
    )

    # Obtain final dataset and store it (push to hub)
    ds.save_to_disk(f"{args.dataset_prefix_path}_processed")


if __name__ == "__main__":
    main()
