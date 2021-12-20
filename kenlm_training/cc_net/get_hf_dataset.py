from typing import Optional

import func_argparse
from datasets import load_dataset
from tqdm import tqdm

from cc_net import text_normalizer


def dl(
    dataset: str,
    output_file: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[str] = None,
    split: Optional[str] = None,
    streaming: bool = True,
    accent: bool = False,
    case: bool = False,
    numbers: bool = True,
    punct: int = 1,
    max_docs: Optional[int] = None,
    seed: int = 0,
    buffer_size: int = 10000,
):
    """Download dataset from the Hugging Face hub."""
    dataset = load_dataset(
        dataset,
        name=name,
        data_dir=data_dir,
        data_files=data_files,
        split=split,
        streaming=streaming,
    )
    dataset_norm = dataset.map(
        lambda x: text_normalizer.normalize(
            x["text"], accent=accent, case=case, numbers=numbers, punct=punct
        )
    )
    dataset_norm = dataset_norm.shuffle(buffer_size=buffer_size, seed=seed)
    count = 0
    with open(output_file, "w") as o:
        with tqdm(total=max_docs) as pbar:
            for doc in dataset_norm:
                count += 1
                doc = doc.rstrip("\n")
                print(doc, file=o)
                if max_docs and count == max_docs:
                    break
                pbar.update(1)


if __name__ == "__main__":
    func_argparse.main(dl)
