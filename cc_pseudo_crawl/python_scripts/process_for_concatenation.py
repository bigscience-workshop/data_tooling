import ast
from argparse import ArgumentParser

from datasets import load_dataset, Dataset

def parse_dataset_with_ratio(elt: str):
    tuple_ = ast.literal_eval(elt)
    assert len(tuple_) == 2
    result = (tuple_[0], float(tuple_[1]))
    assert result[1] <= 1 and result[1] >= 0, "Ratio should be between 0 and 1"
    return result

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--datasets-with_ratio",
        type=lambda x: [parse_dataset_with_ratio(elt) for elt in x.split(",")],
        required=True, help="Dataset name.")

    args = parser.parse_args()

    return args

def collapse_meta_(batch):
    """{"text": str, "meta": str}"""
    # TODO: check that
    columns_not_in_meta = ["text", "html_str"]
    columns_to_collapse = [name for name in batch.keys() if name not in columns_not_in_meta]

    new_batch = {
        "text": batch["text"],
        "meta": [
            str({key: value for key, value in zip(columns_to_collapse, row)})
            for row in zip(*[batch[name] for name in columns_to_collapse])
        ]
    }
    return new_batch

def collapse_meta(ds: Dataset, num_proc):
    """{"text": str, "meta": str}"""
    columns_to_keep = ["text"]
    column_names_to_remove = [name for name in ds.column_names if name not in columns_to_keep]
    return ds.map(collapse_meta_, batched=True, num_proc=num_proc, remove_columns=column_names_to_remove)


def main():
    args = get_args()

    sampled_datasets = {}
    for dataset_path, ratio in args.datasets_with_ratio:
        ds = load_dataset(dataset_path)

        ds = ds.shuffle(seed=42)
        ds = ds[:int(ratio * len(ds))]
        sampled_datasets[dataset_path] = ds




if __name__ == "__main__":
    main()