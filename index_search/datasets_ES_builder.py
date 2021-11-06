import simplejson as json
from datasets.packaged_modules.elasticsearch.elasticsearch import ElasticsearchBuilder

ca_file = "/Users/gdupont/src/github.com/bigscience-workshop/data-tooling/index_search/ca.cert"
with open(
    "/Users/gdupont/src/github.com/bigscience-workshop/data-tooling/index_search/credentials.json"
) as f:
    credentials = json.load(f)

    the_host = credentials["connection"]["https"]["hosts"][0]["hostname"]
    the_port = credentials["connection"]["https"]["hosts"][0]["port"]

    username = credentials["connection"]["https"]["authentication"]["username"]
    psw = credentials["connection"]["https"]["authentication"]["password"]

index_name = "oscar_unshuffled_deduplicated"
oscar_lang_code = "nn"

elasticsearch_builder = ElasticsearchBuilder(
    host=the_host,
    port=the_port,
    es_username=username,
    es_psw=psw,
    ca_file=ca_file,
    es_index_name=index_name,
    es_index_config=None,
    query="mykje arbeid og slit",
)

# elasticsearch_builder = ElasticsearchBuilder(
#     host="localhost",
#     port="9200",
#     es_index_name="oscar_unshuffled_deduplicated",
#     es_index_config=es_index_config,
#     query='"mykje arbeid og slit"'
# )

elasticsearch_builder.download_and_prepare()

oscar_dataset_filtered = elasticsearch_builder.as_dataset()
print(oscar_dataset_filtered.keys())

first_split = next(iter(oscar_dataset_filtered))

for i in range(0, 5):
    print(
        f"- [#{oscar_dataset_filtered[first_split]['id'][i]}] {oscar_dataset_filtered[first_split]['text'][i]}"
    )
