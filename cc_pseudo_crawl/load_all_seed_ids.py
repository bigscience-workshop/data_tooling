import csv
from argparse import ArgumentParser


def get_args():
   parser = ArgumentParser()
   parser.add_argument("--seed-path", type=str, required=True, help="Seed path.")
   parser.add_argument("--seed_index", type=str, required=True, help="Seed index.")
   args = parser.parse_args()

   return args


def main():
   args = get_args()

   with open(f"{__file__}sourcing_sheet_seeds/seeds.csv", "r") as fi:
      data = csv.reader(fi)
      seed_ids = [str(row[0]) for row in data]
      print(seed_ids[args.seed_index])

if __name__ == "__main__":
   main()
