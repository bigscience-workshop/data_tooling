"""Download kenlm models for supported languages (48) from Facebook.

Usage:
    python download_kenlm_models.py --output_path /tmp/

All 48 kenlm language models will be saved under /tmp.
"""

import argparse
import subprocess

from languages_id import langs_id


def download_kenlm_models(output_path: str) -> None:
    supported_kenlm_langs = langs_id["kenlm_id"].dropna().unique()
    # unsupported_kenlm_langs = langs_id.loc[~langs_id.oscar_id.isin(langs_id.kenlm_id)].oscar_id.dropna().unique()
    for lang in supported_kenlm_langs:
        try:
            output_kenlm = subprocess.check_output(
                f"wget http://dl.fbaipublicfiles.com/cc_net/lm/{lang}.arpa.bin -P {output_path}",
                shell=True,
            )
            output_sentence_piece = subprocess.check_output(
                f"wget http://dl.fbaipublicfiles.com/cc_net/lm/{lang}.sp.model -P {output_path}",
                shell=True,
            )
        except:
            print(f"Warning: Download failed for KenLM model for language {lang}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download kenlm models for supported languages."
    )
    parser.add_argument(
        "--output_path", type=str, default="/tmp/", help="Output path to save models."
    )
    args = parser.parse_args()

    download_kenlm_models(output_path=args.output_path)
