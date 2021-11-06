import simplejson as json
from datasets import load_dataset

ca_file = "./ca.cert"

with open("./credentials.json") as f:
    credentials = json.load(f)

    the_host = credentials["connection"]["https"]["hosts"][0]["hostname"]
    the_port = credentials["connection"]["https"]["hosts"][0]["port"]

    username = credentials["connection"]["https"]["authentication"]["username"]
    psw = credentials["connection"]["https"]["authentication"]["password"]

index_name = "oscar_unshuffled_deduplicated"
oscar_lang_code = "nn"

my_dataset = load_dataset(
    "oscar", f"unshuffled_deduplicated_{oscar_lang_code}", split="train"
)

my_dataset.load_elasticsearch_index(
    index_name=index_name,
    host=the_host,
    port=the_port,
    es_username=username,
    es_psw=psw,
    ca_file=ca_file,
    es_index_name=index_name,
    es_index_config=None,
)

print(my_dataset)

K = 10
scores, retrieved = my_dataset.get_nearest_examples(
    index_name, "mykje arbeid og slit", k=K
)

for i in range(0, min(K, len(retrieved))):
    print(f"({i + 1})")
    print(f'\t@{scores[i]:.2f} - {retrieved["id"][i]} => {retrieved["text"][i]} \n')
