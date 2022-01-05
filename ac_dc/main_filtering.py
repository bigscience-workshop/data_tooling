"""Filtering."""

import argparse

from datasets import load_dataset

from filtering import DatasetFiltering


def parseArgs():
    parser = argparse.ArgumentParser(description="Filtering.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="oscar",
        help="Name of the dataset to load.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="unshuffled_deduplicated_af",
        help="Name of the dataset config to pass.",
    )
    parser.add_argument(
        "--data_files",
        type=str,
        default=None,
        help="'load_dataset' returns all files that match the Unix style pattern passed by 'data_files'",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split of the dataset to consider.",
    )
    parser.add_argument(
        "--lang_dataset_id",
        type=str,
        default="af",
        help="ID of the language in which the dataset is written.",
    )
    parser.add_argument(
        "--path_fasttext_model",
        type=str,
        default="ac_dc/lid.176.bin",
        help="Path to the Fasttext model used for language identification.",
    )
    parser.add_argument(
        "--path_sentencepiece_model",
        type=str,
        default="ac_dc/af.sp.model",
        help="Path to the Sentence Piece model used to tokenize text for perplexity scores.",
    )
    parser.add_argument(
        "--path_kenlm_model",
        type=str,
        default="ac_dc/af.arpa.bin",
        help="Path to the KenLM model used to compute perplexity scores.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=2,
        help="Number of processes for multiprocessing.",
    )
    parser.add_argument(
        "--path_dir_save_dataset",
        type=str,
        default="../dataset_filtered/",
        help="Path to the directory where the filtered version of the dataset will be saved.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parseArgs()

    dataset = load_dataset(
        args.dataset_name,
        args.config_name,
        data_files=args.data_files,
        split=args.split,
    )

    dataset_filtering = DatasetFiltering(
        dataset=dataset,
        lang_dataset_id=args.lang_dataset_id,
        path_fasttext_model=args.path_fasttext_model,
        path_sentencepiece_model=args.path_sentencepiece_model,
        path_kenlm_model=args.path_kenlm_model,
        num_proc=args.num_proc,
        path_dir_save_dataset=args.path_dir_save_dataset,
    )
    dataset_filtering.modifying_documents()
    dataset_filtering.filtering()
    dataset_filtering.save_dataset()


if __name__ == "__main__":
    main()
