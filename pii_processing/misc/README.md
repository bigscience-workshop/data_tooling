To test your regex, first install the package and spacy models:
```
git clone https://github.com/bigscience-workshop/data_tooling
cd data_tooling
pip install spacy==3.1  transformers datasets langid faker nltk sentencepiece fsspec tqdm
pip install -r requirements.txt
pip install datasets
python -m nltk.downloader punkt stopwords  wordnet
python -m spacy download en_core_web_lg
python -m spacy download zh_core_web_lg
python -m spacy download pt_core_news_lg
python -m spacy download fr_core_news_lg
python -m spacy download ca_core_news_lg
python -m spacy download es_core_news_lg
cd ..
```

TODO: load dynamic regex file...Then create your regex files under the pii_processing/regex folder of the form ``<initial>_<target_lang>.py``.

Currently: edit the file test_regex.py to add your regex directly under ``__main__``.

Then test your regex on a file of the form ``<target_lang>.jsonl`` which will create a file ``predicted_<target_lang>.jsonl``

```
python data_tooling/pii_processing/misc/test_regex.py -target_lang <target_lang>
```

OR you can import apply_rules from test_regex into your own code
```
from pii_processing.hackathon.test_regex import apply_rules
infile = "<your infile such as en.jsonl>"
outfile = "<your outputfile>"
rulebase = [...] # you can load the rulebases from pii_processing.regex for example
target_lang = "<your lang>"
right, wrong  = apply_rules(infile, outfile, rulebase, target_lang)
```
