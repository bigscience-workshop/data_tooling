import ray
import simplejson as json
from datasets import load_dataset

# ray config on 4 parallel processing
nb_shard = 4
ray.init(num_cpus=nb_shard)

# elasticsearch credentials dev server on IBM cloud
with open("./credentials.json") as f:
    credentials = json.load(f)

    the_host = credentials["connection"]["https"]["hosts"][0]["hostname"]
    the_port = credentials["connection"]["https"]["hosts"][0]["port"]

    username = credentials["connection"]["https"]["authentication"]["username"]
    psw = credentials["connection"]["https"]["authentication"]["password"]

ca_file = "./ca.cert"

oscar_lang_code = "nn"
es_index_config = {
    "settings": {
        "number_of_shards": 1,
        "analysis": {
            "analyzer": {"ngram_analyzer": {"tokenizer": "ngram_tokenizer"}},
            "tokenizer": {
                "ngram_tokenizer": {
                    "type": "ngram",
                    "min_gram": 3,
                    "max_gram": 8,
                    "token_chars": ["letter", "digit"],
                }
            },
        },
    },
    "mappings": {
        "properties": {
            "text": {
                "type": "text",
                "fields": {"hash": {"type": "murmur3"}},
                "analyzer": "ngram_analyzer",
                "similarity": "BM25",
            }
        }
    },
}

index = "oscar_unshuffled_deduplicated"


# @ray.remote
def index_shard(dataset_shard):
    dataset_shard.add_elasticsearch_index(
        column="text",
        index_name=index,
        host=the_host,
        port=the_port,
        es_username=username,
        es_psw=psw,
        ca_file=ca_file,
        es_index_name=index,
        es_index_config=es_index_config,
    )


my_dataset = load_dataset(
    "oscar", f"unshuffled_deduplicated_{oscar_lang_code}", split="train"
)

# single thread indexing
index_shard(my_dataset)

# parallel indexing
# dataset_shards = [my_dataset.shard(nb_shard, i) for i in range(nb_shard)]
# futures = [index_shard.remote(dataset_shard) for dataset_shard in dataset_shards]
# ray.get(futures)

my_dataset.load_elasticsearch_index(
    index_name=index,
    host=the_host,
    port=the_port,
    es_username=username,
    es_psw=psw,
    ca_file=ca_file,
    es_index_name=index,
    es_index_config=es_index_config,
)

K = 10
scores, retrieved = my_dataset.get_nearest_examples(index, "mykje arbeid og slit", k=K)

for i in range(0, min(K, len(retrieved))):
    print(f"({i + 1})")
    print(f'\t@{scores[i]:.2f} - {retrieved["id"][i]} => {retrieved["text"][i]} \n')
