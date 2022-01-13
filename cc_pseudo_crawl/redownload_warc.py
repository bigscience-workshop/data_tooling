from argparse import ArgumentParser

import datasets
datasets.set_caching_enabled(False)
from datasets import load_dataset

from .download_warc import download_warcs

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True, help="Dataset name.")
    parser.add_argument('--num-proc', type=str, required=True, help="Dataset name.")

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    ds = load_dataset(args.dataset_path)

    download_warcs(ds, args.dataset_path, num_proc=args.num_proc)

if __name__ == "__main__":
    main()