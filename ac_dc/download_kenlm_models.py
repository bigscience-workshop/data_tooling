"""Download kenlm models for supported languages (48) from Facebook.

Usage:
    python download_kenlm_models.py --output_path /tmp/

All 48 kenlm language models will be saved under /tmp.
"""

import argparse
import os

from languages_id import langs_id


def download_kenlm_models(output_path: str) -> None:
    supported_kenlm_langs = langs_id["kenlm_id"].dropna().unique()
    for lang in supported_kenlm_langs[1:]:
        os.system(
            f"wget http://dl.fbaipublicfiles.com/cc_net/lm/{lang}.arpa.bin -P {output_path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download kenlm models for supported languages."
    )
    parser.add_argument(
        "--output_path", type=str, default="/tmp/", help="Output path to save models."
    )
    args = parser.parse_args()

    download_kenlm_models(output_path=args.output_path)
