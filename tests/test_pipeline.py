from ac_dc.badwords import badwords
from ac_dc.config import PipelineConfig
from ac_dc.basic_pipeline import BasicPipeline
from tqdm import tqdm
import gzip
from tqdm import tqdm as tqdm


def test_basic_pipeline():
    lang_oscar_id = "en"
    path_model_fasttext = "/tmp/lid.176.bin"
    path_oscar_file = "/tmp/en_part_1.txt.gz"

    parameters_filtering_en = {
        "cond_remove_words_with_incorrect_substrings": True,
        "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
        "cond_remove_long_words": True,
        "length_word_cutoff": 25,
        "cond_check_empty": True,
        "strip_characters": "' 0123456789¯_%$§½¼¾×|†~\"—±′–'°−{}[]·'?,./<>!@#^&*()+-‑=:;，：,...�`→¶.…‘’”",
        "cond_check_special_characters": True,
        "special_characters": "' 0123456789¯_%$§½¼¾×|†~\"—±′–'°−{}[]·'?,./<>!@#^&*()+-‑=:;，：,...�`→¶.…‘’”",
        "special_characters_cutoff": 0.4,
        "cond_check_stopwords": True,
        "stopwords_cutoff": 0.4,
        "cond_check_badwords": True,
        "badwords_cutoff": 0.4,
        "badwords": badwords[lang_oscar_id],
        "cond_check_lang_id": True,
        "path_model_fasttext": path_model_fasttext,
        "lang_oscar_id": lang_oscar_id,
        "lang_id_cutoff": 0.8,
    }

    config = PipelineConfig("en", **parameters_filtering_en)
    pipeline = BasicPipeline(config)

    dropped_records = path_oscar_file.replace(".txt.gz", "") + ".sample_dropped.txt"
    kept_records = path_oscar_file.replace(".txt.gz", "") + ".sample_kept.txt"

    with open(dropped_records, "w", encoding="utf-8") as fd:
        with open(kept_records, "w", encoding="utf-8") as fk:
            with gzip.open(path_oscar_file, "rb") as f2:
                for id_, line in enumerate(tqdm(f2)):
                    line = pipeline.normalize(line.decode().strip())
                    if pipeline.filter(line):
                        fk.write(line + "\n\n")
                    else:
                        fd.write(line + "\n\n")
                    if id_ > 1000:
                        break
    print("Dropped:", dropped_records)
    print("Kept:", kept_records)
    assert True
