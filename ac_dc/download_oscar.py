"""Download Oscar v1.

Usage:
    python download_oscar.py --output_path /tmp/

The Oscar dataset will be saved under /tmp.
"""

import argparse
import subprocess
import fsspec

from languages_id import langs_id


def get_oscar_urls(language, shuffled="unshuffled", deduplicated="deduplicated"):
    _BASE_DATA_URL_FORMAT_STR = "https://s3.amazonaws.com/datasets.huggingface.co/oscar/1.0/{shuffled}/{deduplicated}/{language}/"
    _BASE_CHECKSUM_FILE_NAME = "{language}_sha256.txt"
    base_data_url = _BASE_DATA_URL_FORMAT_STR.format(
        shuffled=shuffled, language=language, deduplicated=deduplicated
    )
    checksum_url = base_data_url + _BASE_CHECKSUM_FILE_NAME.format(language=language)
    with fsspec.open(checksum_url, encoding="utf-8") as f:
        data_filenames = [line.decode().split("\t")[0] for line in f if line]
    return [base_data_url + data_filename for data_filename in data_filenames]


def download_oscar(output_path: str) -> None:
    supported_langs = langs_id["oscar_id"]
    for lang in supported_langs:
        try:
            urls = get_oscar_urls(lang)
            for url in urls:
                output = subprocess.check_output(
                    f"wget {url} -P {output_path}",
                    shell=True,
                )
        except:
            print(f"Warning: Download failed or incomplete for language {lang}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Oscar v1 dataset for supported languages."
    )
    parser.add_argument(
        "--output_path", type=str, default="/tmp/", help="Output path to save models."
    )
    args = parser.parse_args()

    download_oscar(output_path=args.output_path)
