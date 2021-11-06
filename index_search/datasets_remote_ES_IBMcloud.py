import base64
import ssl

import simplejson as json
from elasticsearch import Elasticsearch

with open("./credentials.json") as f:
    credentials = json.load(f)

    host = credentials["connection"]["https"]["hosts"][0]["hostname"]
    port = credentials["connection"]["https"]["hosts"][0]["port"]

    es_username = credentials["connection"]["https"]["authentication"]["username"]
    es_psw = credentials["connection"]["https"]["authentication"]["password"]

    ca_cert = base64.b64decode(
        credentials["connection"]["https"]["certificate"]["certificate_base64"]
    )
    # context = ssl.create_default_context()
    # context.verify_mode = ssl.CERT_REQUIRED
    # context.load_verify_locations(cadata=ca_cert)

context = ssl.create_default_context(cafile="./ca.cert")

server_url = (
    ("https" if context is not None else "http") + "://" + host + ":" + str(port)
)

es = Elasticsearch([server_url], http_auth=(es_username, es_psw), ssl_context=context)

print(f"ES info {json.dumps(es.info(), indent=4 * ' ')}")

# index_get_response = es.indices.get(index='oscar_unshuffled_deduplicated')
# print(json.dumps(index_get_response, indent=4 * ' '))

delete_response = es.indices.delete(index="oscar_unshuffled_deduplicated")
print(json.dumps(delete_response, indent=4 * " "))
