from datasets import load_dataset

import ray


ray.init()

language_stop_map = {
    "ara": "_arabic_",
    "hye": "_armenian_",
    "eus": "_basque_",
    "ben": "_bengali_",
    # "por": "_brazilian_", # (Brazilian Portuguese)
    "bul": "_bulgarian_",
    "cat": "_catalan_",
    "zho": "_cjk_",  # (Chinese, Japanese, and Korean)
    "jpn": "_cjk_",
    "kor": "_cjk_",
    "ces": "_czech_",
    "dan": "_danish_",
    "nld": "_dutch_",
    "eng": "_english_",
    "est": "_estonian_",
    "fin": "_finnish_",
    "fra": "_french_",
    "glg": "_galician_",
    "deu": "_german_",
    "ell": "_greek_",
    "hin": "_hindi_",
    "hun": "_hungarian_",
    "ind": "_indonesian_",
    "gle": "_irish_",
    "ita": "_italian_",
    "lav": "_latvian_",
    "lit": "_lithuanian_",
    "nor": "_norwegian_",
    "fas": "_persian_",
    "por": "_portuguese_",
    "ron": "_romanian_",
    "rus": "_russian_",
    "ckb": "_sorani_",
    "spa": "_spanish_",
    "swe": "_swedish_",
    "tha": "_thai_",
    "tur": "_turkish_"
}

es_index_config = \
    {
        "settings": {
            "number_of_shards": 1,
            "analysis": {"analyzer": {"stop_standard": {"type": "standard", " stopwords": language_stop_map["cat"]}}},
        },
        "mappings": {"properties": {"text": {"type": "text", "analyzer": "standard", "similarity": "BM25"}}},
    }


@ray.remote
def index_shard(dataset, shards, shard_id):
    dataset_shard = dataset.shard(shards, shard_id)
    dataset_shard.add_elasticsearch_index("text", host="localhost",
                                          port="9200",
                                          es_index_name='big_science_oscar_unshuffled_original_ca',
                                          es_index_config=es_index_config)
    return 0


my_dataset = load_dataset('oscar', 'unshuffled_original_ca', split='train')

nb_shard = 4
futures = [index_shard.remote(my_dataset, nb_shard, i) for i in range(nb_shard)]
print(ray.get(futures))

K = 10
scores, retrieved_examples = my_dataset.get_nearest_examples("text", "esqueixada", k=K)

for i in range(0, K):
    print(f'({i+1}) @{scores[i]:.2f} - {retrieved_examples["id"][i]} => {retrieved_examples["text"][i]} \n')
