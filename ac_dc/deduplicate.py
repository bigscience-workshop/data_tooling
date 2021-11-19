"""Creating simhashes and removing near duplicates."""

import os
import logging
import re
import datetime
from collections import defaultdict
from typing import List, Tuple, Optional
from multiprocessing import Manager, Queue, Process, cpu_count

import cloudpickle
import networkx as nx
import typer
from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets
from simhash import Simhash, SimhashIndex

app = typer.Typer()
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)


def create_shingles(text: str, window: int = 4) -> List[str]:
    """
    Create shingles/ngrams from the given text.

    Parameters
    ----------
    text : str
        Input text string
    window : int, optional
        The size of the window, by default 4

    Returns
    -------
    List[str]
        List of shingles

    Examples
    --------
    >>> create_shingles("This is a test message", window=3)
    ['thi', 'his', 'isi', 'sis', 'isa', 'sat', 'ate', 'tes', 'est', 'stm', 'tme', 'mes', 'ess', 'ssa', 'sag', 'age']
    """
    if len(text) <= window:
        return [text]

    text = re.sub(r"[^\w]+", "", text.lower())
    return [text[i : i + window] for i in range(len(text) - window + 1)]


def check_num_proc(num_proc: int = -1):

    maximum: int = cpu_count()
    if num_proc > maximum:
        raise ValueError(
            f"{num_proc} exceeds the maximum number ({maximum}) of processors"
        )

    if num_proc == -1:
        num_proc = maximum
    else:
        logger.warning(f"Using {num_proc} out of {maximum} can be slow")

    return num_proc


def shard(
    ds: DatasetDict,
    split: str,
    num_shards: int,
    idx: int,
    output_dir: str,
):
    ds[split].shard(num_shards=num_shards, index=idx).to_json(
        os.path.join(output_dir, f"sharded_{idx:05d}.jsonl"),
        orient="records",
        lines=True,
        force_ascii=False,
    )


@app.command()
def create_shards(
    output_dir: str,
    num_shards: int,
    path: str = typer.Option(
        "mhtoin/register_oscar", help="Path or name of the dataset"
    ),
    name: Optional[str] = typer.Option(
        None, help="Defining the name of the dataset configuration"
    ),
    data_dir: Optional[str] = typer.Option(
        None, help="Defining the data_dir of the dataset configuration"
    ),
    split: Optional[str] = typer.Option(None, help="Which split of the data to load"),
):
    """
    Shard a dataset into multiple parts.

    Parameters
    ----------
    output_dir : str
        Directory path for all the subset files
    num_shards : int
        Number of shards to use
    path : str, optional
        Path to the dataset configuration, by default typer.Option( "mhtoin/register_oscar", help="Path or name of the dataset" )
    name : Optional[str], optional
        Name of the dataset configuration, by default typer.Option( None, help="Defining the name of the dataset configuration" )
    data_dir : Optional[str], optional
        Local data directory, by default typer.Option( None, help="Defining the datadata_dir of the dataset configuration" )
    split : Optional[str], optional
        The split of the dataset configuration, by default typer.Option(None, help="Which split of the data to load")
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    ds = load_dataset(path, name, data_dir=data_dir, use_auth_token=True)

    queue = Queue()
    processes = [
        Process(target=shard, args=(ds, split, num_shards, idx, output_dir))
        for idx in range(num_shards)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


@app.command()
def build_hashes(
    output_dir: str,
    path: str = typer.Option(
        "mhtoin/register_oscar", help="Path or name of the dataset"
    ),
    name: Optional[str] = typer.Option(
        None, help="Defining the name of the dataset configuration"
    ),
    data_dir: Optional[str] = typer.Option(
        None, help="Defining the data_dir of the dataset configuration"
    ),
    data_files: Optional[List[str]] = typer.Option(
        None, help="Path(s) to source data file(s)"
    ),
    split: Optional[str] = typer.Option(None, help="Which split of the data to load"),
    shingle_size: int = typer.Option(4, help="Size of the generated shingles"),
    num_proc: int = typer.Option(-1, help="Number of processes to use"),
    text_column_name: Optional[str] = typer.Option(
        "text", help="Column name of the text"
    ),
):
    """
    Create a single dataset with an extra `hash` column from all data files.

    The time complexity for this function call is rounghly O(N * L // C) where N is the number
    of records in the data, L is the average length of each record text and C is the number of processes.
    The addition of the hashes should be insignificant in size compared to the original data (12MB vs 800MB for two English shards).

    Example:

    ```bash
    python ac_dc/deduplicate.py build-hashes "cache/en_hashes_00001" --data-files "en/en_00001.jsonl.gz" --data-files "en/en_00002.jsonl.gz" --path "mhtoin/register_oscar" --split "train"
    ```

    This generates a dataset `cache/en_hashes_00001` from two shards `en/en_00001.jsonl.gz` and `en/en_00002.jsonl.gz` in `mhtoin/register_oscar`.

    Parameters
    ----------
    output_dir : str
        Output directory of the new data
    path : str, optional
        Path or name of the dataset
    name : Optional[str], optional
        Defining the name of the dataset configuration
    data_dir : Optional[str], optional
        Defining the data_dir of the dataset configuration
    data_files : Optional[List[str]], optional
        Path(s) to source data file(s)
    split : Optional[str], optional
        Which split of the data to load
    shingle_size : int, optional
        Size of the generated shingles
    num_proc : int, optional
        Number of processes to use
    text_column_name : Optional[str], optional
        Column name of the text
    """
    num_proc = check_num_proc(num_proc)
    ds = load_dataset(path=path, name=name, data_files=data_files, data_dir=data_dir)

    def process(record):
        return {
            "hash": str(
                Simhash(create_shingles(record[text_column_name], shingle_size)).value
            )
        }

    splits = [split] if split is not None else list(ds.keys())
    for s in splits:
        ds[s] = ds[s].map(process, num_proc=num_proc)
    ds.save_to_disk(output_dir)


@app.command()
def build_index(
    output_file: str,
    data_dirs: List[str],
    split: Optional[str] = typer.Option(None, help="Which split of the data to load"),
    threshold: int = typer.Option(
        4,
        help="Hamming fingerprint distance threshold, below which data are considered to be duplicates",
    ),
    num_proc: int = typer.Option(-1, help="Number of processes to use"),
):
    """
    Merging all hashes and build an index. The time complexity and space complexity for this function is at least O(N).
    Building the index is not paralleled in this implementation since the index needs access to all hashes.

    Example:

    ```bash
    python deduplicate.py build-index "cache/en_simhash_index.pkl" "cache/en_hashes_00001" --split "train"
    ```

    This builds an index to be stored at `cache/en_simhash_index.pkl` from `cache/en_hashes_00001`

    Parameters
    ----------
    output_file : str
        Output path for the index file
    data_dirs : List[str]
        Dataset directories with hashes to build the index from
    split : Optional[str], optional
        Which split of the data to load
    threshold : int, optional
        Hamming fingerprint distance threshold, below which data are considered to be duplicates, by default 4
    num_proc : int, optional
        Number of processes to use
    """
    num_proc = check_num_proc(num_proc)

    # Shared object
    manager = Manager()
    hashes: List[Tuple[str, Simhash]] = manager.list()

    def process(record):
        hashes.append((record["id"], Simhash(int(record["hash"]))))
        return

    for dir in data_dirs:
        ds = load_from_disk(dir)
        splits = [split] if split is not None else list(ds.keys())
        if split is None:
            logger.warning(
                f"Using all splits to build the index, please make sure the `id` is unique globally"
            )
        for s in splits:
            ds[s].map(process, num_proc=num_proc)

    index = SimhashIndex(hashes, k=threshold, log=logger)
    with open(output_file, "wb") as o:
        o.write(cloudpickle.dumps(index))


@app.command()
def find_duplicates(
    data_dirs: List[str],
    index_file: str,
    split: Optional[str] = typer.Option(None, help="Which split of the data to load"),
    num_proc: int = typer.Option(-1, help="Number of processes to use"),
):
    """
    Find duplicates for given datasets. For each dataset directory `d`, it outputs a `d_duplicates` directory
    with a new `duplicates` column, containing all the duplicate indices.

    The time complexity is O(N * B // C) where B is a large number (the size of a bucket) and C is the number of processes.

    Example:

    ```bash
    python deduplicate.py find-duplicates "cache/en_hashes_00001" "cache/en_simhash_index.pkl" --split "train"
    ```

    This finds all duplicates in `cache/en_hashes_00001` with `cache/en_simhash_index.pkl`. It should outputs a directory named
    `cache/en_hashes_00001_duplicates`.

    Parameters
    ----------
    data_dirs : List[str]
        List of dataset directories to find duplicates
    index_file : str
        Path to the index file
    split : Optional[str], optional
        Which split of the data to load
    num_proc : int, optional
        Number of processes to use
    """
    num_proc = check_num_proc(num_proc)

    with open(index_file, "rb") as i:
        index = cloudpickle.loads(i.read())

    def process(record):
        nonlocal index
        duplicates = list(map(str, index.get_near_dups(Simhash(int(record["hash"])))))
        # arrow forces the features to be the same data type list<item: string>>
        # so add a dummy string here when this is used on two different datasets
        duplicates = duplicates if duplicates else [""]
        return {"duplicates": duplicates}

    for dir in data_dirs:
        ds = load_from_disk(dir)
        splits = [split] if split is not None else list(ds.keys())
        for s in splits:
            ds[s] = ds[s].map(process, num_proc=num_proc)
            logger.info(
                f"Found {len(ds[s].filter(lambda x: len(x['duplicates']) > 1))} duplicates in {dir}"
            )
        ds.save_to_disk(dir.rstrip("/") + "_duplicates")


@app.command()
def remove_duplicates(
    data_dirs: List[str],
    split: Optional[str] = typer.Option(None, help="Which split of the data to load"),
    num_proc: int = typer.Option(-1, help="Number of processes to use"),
):
    """
    Remove duplicates based on the `duplicates` column by finding the connected components and only keep the first occurrence.
    For each data directory `d`, it outputs a `d_deduplicated` directory.

    Example:

    ```bash
    python deduplicate.py remove-duplicates "cache/en_hashes_00001_duplicates" --split "train"
    ```
    This removes all duplicates from `cache/en_hashes_00001_duplicates` and create
    a deduplicated version in `cache/en_hashes_00001_deduplicated`.

    Parameters
    ----------
    data_dirs : List[str]
        List of data directories to remove duplicates from
    split : Optional[str], optional
        Which split of the data to load
    num_proc : int, optional
        Number of processes to use
    """
    num_proc = check_num_proc(num_proc)

    # a and b are connected if they are duplicates
    G = nx.Graph()
    manager = Manager()
    edges = manager.list()

    def process(record):
        for dup in record["duplicates"]:
            if str(record["id"]) == dup:
                continue
            edges.append((str(record["id"]), dup))

    for dir in data_dirs:
        ds = load_from_disk(dir)
        splits = [split] if split is not None else list(ds.keys())
        for s in splits:
            ds[s].map(process, num_proc=num_proc)

    flags = defaultdict(lambda: False)
    for x, y in edges:
        G.add_edge(x, y)

    for c in nx.connected_components(G):
        for n in c:
            flags[n] = False
        flags[c.pop()] = True

    for dir in data_dirs:
        ds = load_from_disk(dir)
        splits = [split] if split is not None else list(ds.keys())
        for s in splits:
            ds[s] = ds[s].filter(lambda x: flags.get(str(x["id"]), True))
        ds.save_to_disk(dir.rstrip("/").replace("_duplicates", "_deduplicated"))


@app.command()
def merge_meta(
    data_dirs: List[str] = typer.Option(None, help="Source data to add metadata in"),
    meta_data_dirs: List[str] = typer.Option(
        None, help="Reference data to extract metadata from"
    ),
    split: Optional[str] = typer.Option(None, help="Which split of the data to load"),
    num_proc: int = typer.Option(-1, help="Number of processes to use"),
):
    """
    Extracting metadata feature from `meta_data_dirs` and merging into data in `data_dirs`
    For each data directory `d`, it outputs a `d_with_meta` directory.

    see examples/merge.sh for an example

    Parameters
    ----------
    data_dirs : List[str]
        List of data directories to add metadata to
    meta_data_dirs : List[str]
        List of data directories to extract metadata from
    split : Optional[str], optional
        Which split of the data to load
    num_proc : int, optional
        Number of processes to use
    """
    num_proc = check_num_proc(num_proc)
    manager = Manager()
    meta_data = manager.dict()

    # Collect all metadata
    def process_meta(record):
        meta_data[str(record["id"])] = record["meta"]

    for dir in meta_data_dirs:
        ds = load_from_disk(dir)
        splits = [split] if split is not None else list(ds.keys())
        for s in splits:
            ds[s].map(process_meta, num_proc=num_proc)

    # add meta into the record
    def merge(record):

        metadata = {
            "headers": {
                "warc-record-id": "",
                "warc-date": datetime.datetime(1970, 1, 1),
                "content-type": "",
                "content-length": -1,
                "warc-type": "",
                "warc-identified-content-language": "",
                "warc-refers-to": "",
                "warc-target-uri": "",
                "warc-block-digest": "",
            },
            "offset": -1,
            "nb_sentences": -1,
        }

        if not record["duplicates"]:
            return {"meta": metadata}

        for dup in record["duplicates"]:
            if dup in meta_data:
                metadata = meta_data[dup]
                break

        return {"meta": metadata}

    for dir in data_dirs:
        ds = load_from_disk(dir)
        splits = [split] if split is not None else list(ds.keys())
        for s in splits:
            ds[s] = ds[s].map(merge, num_proc=num_proc)
        ds.save_to_disk(dir.rstrip("/").replace("_duplicates", "_with_meta"))


@app.command()
def merge_shards(
    output_dir: str,
    data_dirs: list[str],
    split: Optional[str] = typer.Option(None, help="Which split of the data to load"),
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    ds = []
    for dir in data_dirs:
        ds.append(load_from_disk(dir)[split])

    DatasetDict({split: concatenate_datasets(ds)}).save_to_disk(output_dir)


if __name__ == "__main__":

    app()
