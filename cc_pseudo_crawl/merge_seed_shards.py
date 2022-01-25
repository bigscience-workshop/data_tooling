from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset-shard", type=str, required=True, help="Dataset directory containing all shards")
    parser.add_argument("--save-prefix-path", type=str, required=True, help="Where to save the dataset.")
    parser.add_argument("--num-proc", type=int, default=1, help="Number of procs use for preprocessing.")
    args = parser.parse_args()

    args.dataset_dir = Path(args.dataset_dir)
    args.save_dir = Path(args.save_dir)
    return args

def main():
    args = get_args()

if __name__ == "__main__":
    main()