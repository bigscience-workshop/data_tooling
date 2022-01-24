import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import hydra
import wandb
from datasets import Dataset, Features, config, load_dataset, load_from_disk
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from bsmetadata.preprocessing_utils import (
    DatasourcePreprocessor,
    EntityPreprocessor,
    ErrorWrapperPreprocessor,
    GenerationLengthPreprocessor,
    HtmlPreprocessor,
    MetadataPreprocessor,
    TimestampPreprocessor,
    UrlPreprocessor,
    WebsiteDescPreprocessor,
)
from bsmetadata.train import show_help


logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    task_id: int = field(metadata={"help": "The id of the task"})
    out_dir: str = field(metadata={"help": "where to save the resulting dataset."})
    num_files_to_process: int = field(metadata={"help": "the number of files to process"})
    path_wiki_db: Optional[str] = field(
        metadata={"help": "The path to the wikipedia database file necessary for the website descriptions"}
    )
    entity_path_data_dir: Optional[str] = field(
        metadata={
            "help": "The path to the directory containing the directories `ed-wiki-2019`, `generic` and `wiki_2019` "
        }
    )
    path_or_url_flair_ner_model: Optional[str] = field(
        default=None, metadata={"help": "TThe path or name of the flair ner model to use to preprocess entities"}
    )
    metadata_to_include: Optional[list] = field(
        default_factory=lambda: ["website_description", "entity", "timestamp"],
        metadata={"help": "The list of metadata to extract"},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)"}
    )
    overwrite_cache: Optional[bool] = field(
        default=False, metadata={"help": "Whether the local cache containing datasets should be overwritten."}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3?"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    map_batch_size: Optional[int] = field(
        default=1,
        metadata={
            "help": "This is the size of the batch size that will be used for the mapping operation when generating"
            " the dataset. If you are using `with_metadata` the recommended batch size is 1.."
        },
    )
    project_name: str = field(default="metadata_lm_exploration", metadata={"help": "The project name."})
    save_batch_size: int = field(
        default=datasets.config.DEFAULT_MAX_BATCH_SIZE,
        metadata={"help": " Size of the batch to load in memory and write at once."},
    )
    use_load_from_disk: bool = field(
        default=False,
        metadata={
            "help": "If false, the program will load the dataset with `load_dataset` and if false, it will load it "
            "with `load_from_disk`."
        },
    )


class Logger:
    def __init__(self, *args, **kwargs):
        self.run = wandb.init(*args, **kwargs)

    def log(self, dic):
        wandb.log(dic)

    def close(self):
        wandb.finish()


cs = ConfigStore.instance()
cs.store(name="preprocessing_config", node=PreprocessingConfig)


col_html = "html_str"
col_url = "url"
col_to_store_text = "text"
col_to_store_head = "html_head"
col_to_store_footer = "html_footer"
col_to_store_title = "html_title"
col_to_store_metadata_html = "metadata_html"
col_to_store_metadata_url = "metadata_url"
col_to_store_metadata_timestamp = "metadata_timestamp"
col_to_store_metadata_website_desc = "metadata_website_desc"
col_to_store_metadata_entities = "metadata_entity"
col_to_store_metadata_generation_length_text = "metadata_generation_length_text"
col_to_store_metadata_generation_length_sentence = "metadata_generation_length_sentence"
col_to_store_metadata_datasource = "metadata_generation_datasource"


@hydra.main(config_name="preprocessing_config")
def main(args: PreprocessingConfig) -> None:  # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    config_dict = OmegaConf.to_container(args)
    metrics_logger = Logger(project=args.project_name, config=config_dict)

    logger.info("Initialize the preprocessors:")
    if "html" in args.metadata_to_include:
        logger.info("   Html...")
        _html_processor = HtmlPreprocessor(
            col_to_store_metadata=col_to_store_metadata_html,
            col_to_store_text=col_to_store_text,
            col_html=col_html,
            col_to_store_footer=col_to_store_footer,
            col_to_store_head=col_to_store_head,
            col_to_store_title=col_to_store_title,
        )
        html_processor = ErrorWrapperPreprocessor(
            metadata_preprocessor=_html_processor,
            output_keys={
                col_to_store_metadata_html: [],
                col_to_store_text: "",
                col_to_store_footer: [],
                col_to_store_head: [],
                col_to_store_title: [],
            },
        )

    if "url" in args.metadata_to_include:
        logger.info("   Url...")
        url_processor = UrlPreprocessor(col_to_store_metadata=col_to_store_metadata_url, col_url=col_url)

    if "timestamp" in args.metadata_to_include:
        logger.info("   Timestamp...")
        timestamp_processor = TimestampPreprocessor(
            col_to_store_metadata=col_to_store_metadata_timestamp, col_metadata_url=col_to_store_metadata_url
        )

    if "website_description" in args.metadata_to_include:
        logger.info("   Website description...")
        website_processor = WebsiteDescPreprocessor(
            col_to_store_metadata=col_to_store_metadata_website_desc,
            col_metadata_url=col_to_store_metadata_url,
            path_wiki_db=args.path_wiki_db,
        )

    if "entity" in args.metadata_to_include:
        logger.info("   Entity...")
        entity_processor = EntityPreprocessor(
            base_url=args.entity_path_data_dir,
            path_wiki_db=args.path_wiki_db,
            path_or_url_flair_ner_model=args.path_or_url_flair_ner_model,
            col_to_store_metadata=col_to_store_metadata_entities,
            col_text=col_to_store_text,
        )

    if "generation_length_text" in args.metadata_to_include:
        logger.info("   Generation length text...")
        generation_length_preprocessor_text = GenerationLengthPreprocessor(
            mode="text", col_to_store_metadata=col_to_store_metadata_generation_length_text
        )

    if "generation_length_sentence" in args.metadata_to_include:
        logger.info("   Generation length sentence...")
        generation_length_preprocessor_sentence = GenerationLengthPreprocessor(
            mode="sentence", col_to_store_metadata=col_to_store_metadata_generation_length_sentence
        )

    if "datasource" in args.metadata_to_include:
        logger.info("   Datasource...")
        datasource_preprocessor = DatasourcePreprocessor(
            col_to_store_metadata=col_to_store_metadata_datasource, col_url="url"
        )
    logger.info("Processors initialization finished")

    poss_files = sorted(os.listdir(args.dataset_name))

    if args.use_load_from_disk:
        poss_files = [file_name for file_name in poss_files if file_name.startswith("c4-en-html")]
    else:
        poss_files = [
            file_name
            for file_name in poss_files
            if (file_name.endswith("jsonl.gz") or file_name.endswith("jsonl")) and file_name.startswith("c4-en-html")
        ]

    def process_file(file_name: str):

        logger.info(config.HF_DATASETS_CACHE)
        processing_name = (
            "-".join(args.metadata_to_include) if args.metadata_to_include is not None else "full-process"
        )
        metrics_logger.log({processing_name: 0})

        metrics_logger.log({"load_dataset": 0})
        if args.use_load_from_disk:
            dataset_name = os.path.join(args.dataset_name, file_name)
            logger.info(f"Loading the dataset {dataset_name} with `load_from_disk`")
            ds = load_from_disk(dataset_name)
        else:
            data_files = {"file": file_name}
            logger.info(
                "Loading a dataset with `load_dataset`"
                f"{args.dataset_name}, {args.dataset_config_name}, data_files={data_files}, cache_dir={args.cache_dir},"
            )
            ds = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                data_files=data_files,
                cache_dir=args.cache_dir,
                keep_in_memory=False,
                download_mode="force_redownload",
            )["file"]

        metrics_logger.log({"load_dataset": 1})

        features_dict = dict(ds.features)
        logger.info(f"the initial features of the dataset are: {features_dict}")

        def apply_processor(ds: Dataset, processor: MetadataPreprocessor, remove_columns=None) -> Dataset:
            for col_name, feature_type in processor.new_columns_minimal_features.items():
                assert col_name not in features_dict
                features_dict[col_name] = feature_type
            extraction_name = processor.__class__.__name__

            logger.info(f"Start {extraction_name}")
            metrics_logger.log({extraction_name: 0})
            ds = ds.map(
                processor.preprocess,
                batched=True,
                batch_size=args.map_batch_size,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Running {extraction_name} on dataset",
                features=Features(features_dict),
                remove_columns=remove_columns,
            )
            metrics_logger.log({extraction_name: 1})
            logger.info(f"End {extraction_name}")
            return ds

        if "html" in args.metadata_to_include:
            ds = apply_processor(ds=ds, processor=html_processor)

        if "url" in args.metadata_to_include:
            ds = apply_processor(ds=ds, processor=url_processor)

        if "timestamp" in args.metadata_to_include:
            ds = apply_processor(ds=ds, processor=timestamp_processor)

        if "website_description" in args.metadata_to_include:
            ds = apply_processor(ds=ds, processor=website_processor)

        if "entity" in args.metadata_to_include:
            ds = apply_processor(ds=ds, processor=entity_processor)

        if "generation_length_text" in args.metadata_to_include:
            ds = apply_processor(ds=ds, processor=generation_length_preprocessor_text)

        if "generation_length_sentence" in args.metadata_to_include:
            ds = apply_processor(ds=ds, processor=generation_length_preprocessor_sentence)

        if "datasource" in args.metadata_to_include:
            ds = apply_processor(ds=ds, processor=datasource_preprocessor)

        if file_name.endswith(".jsonl.gz"):
            out_file_name = file_name[: -len(".jsonl.gz")]
        elif file_name.endswith(".jsonl"):
            out_file_name = file_name[: -len(".jsonl")]
        else:
            out_file_name = file_name
        out_file_name_tmp = f"tmp-{out_file_name}"

        saving_path = os.path.join(args.out_dir, out_file_name)
        saving_path_tmp = os.path.join(args.out_dir, out_file_name_tmp)

        logger.info(f"Save resulting dataset {ds} at {saving_path_tmp}")
        metrics_logger.log({"save_result": 0})
        ds.save_to_disk(saving_path_tmp)
        metrics_logger.log({"save_result": 1})
        logger.info(f"Moving the saved dataset to {saving_path}")
        subprocess.run(["mv", saving_path_tmp, saving_path])
        logger.info(f"Processing of {file_name} ended successfully")
        metrics_logger.log({processing_name: 1})

    for file_name in poss_files[
        args.task_id * args.num_files_to_process : args.task_id * args.num_files_to_process + args.num_files_to_process
    ]:
        logger.info(f"Start to process {file_name}")
        process_file(file_name=file_name)

    metrics_logger.close()


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        sys.exit()
    main()
