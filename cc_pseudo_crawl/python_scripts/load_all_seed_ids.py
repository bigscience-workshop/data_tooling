import csv
from argparse import ArgumentParser
from pathlib import Path


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--seed-path",
        type=str,
        required=True,
        help="Seed full path. e.g. 'xxx/seeds.csv'",
    )
    parser.add_argument("--seed-index", type=int, required=True, help="Seed index.")
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    with open(args.seed_path, "r") as fi:
        data = csv.reader(fi)
        # First line is all the headers that we remove.
        seed_ids = [row[0] for row_id, row in enumerate(data) if row_id > 0]
        print(seed_ids[args.seed_index])


if __name__ == "__main__":
    main()
