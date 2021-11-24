from functools import partial

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from perplexity_lenses import REGISTRY_DATASET
from perplexity_lenses.perplexity import KenlmModel


def hub_dataset_to_dataframe(
    path: str,
    name: str,
    split: str,
    sample: int,
    text_column: str,
    model: KenlmModel,
    seed: int = 0,
    doc_type: str = "Whole document",
) -> pd.DataFrame:
    load_dataset_fn = partial(load_dataset, path=path)
    if name:
        load_dataset_fn = partial(load_dataset_fn, name=name)
        # Special case for the registry dataset
        if path == REGISTRY_DATASET:
            load_dataset_fn = partial(load_dataset_fn, data_files=f"{name}/*")
    if split:
        load_dataset_fn = partial(load_dataset_fn, split=split)
    dataset = load_dataset_fn(streaming=True).shuffle(buffer_size=10000, seed=seed)
    if doc_type.lower() == "sentence":
        dataset = dataset.map(
            lambda x: [
                {
                    text_column: sentence,
                    "perplexity": model.get_perplexity(sentence),
                    "label": x.get("labels", [])[0]
                    if len(x.get("labels", [])) > 0
                    else "NONE",  # Special case for registry dataset
                }
                for sentence in x[text_column].split("\n")
            ]
        )
    else:
        dataset = dataset.map(
            lambda x: {
                text_column: x[text_column],
                "perplexity": model.get_perplexity(x[text_column]),
                "label": x.get("labels", [])[0]
                if len(x.get("labels", [])) > 0
                else "NONE",  # Special case for registry dataset
            }
        )
    instances = []
    count = 0
    for instance in tqdm(dataset, total=sample):
        if isinstance(instance, list):
            for sentence in instance:
                instances.append(sentence)
                count += 1
                if count == sample:
                    break
        else:
            instances.append(instance)
            count += 1
        if count == sample:
            break
    return pd.DataFrame(instances)


def documents_df_to_sentences_df(
    df: pd.DataFrame, text_column: str, sample: int, seed: int = 0
):
    df_sentences = pd.DataFrame(
        {
            text_column: np.array(
                df[text_column].map(lambda x: x.split("\n")).values.tolist()
            ).flatten()
        }
    )
    return df_sentences.sample(min(sample, df_sentences.shape[0]), random_state=seed)
