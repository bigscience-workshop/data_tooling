---
title: Perplexity Lenses
emoji: 🌸
colorFrom: pink
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---

# Installation:
Requires Python >= 3.7 and < 3.10
```
pip install .
```
Or with [poetry](https://python-poetry.org/)
```
poetry install
```

# Web App:
The app is hosted [here](https://huggingface.co/spaces/edugp/perplexity-lenses). To run it locally:
```
python -m streamlit run app.py
```

# CLI:
The CLI with no arguments defaults to running mc4 in Spanish.
For full usage:
```
python cli.py --help
```
Example: Running on 1000 sentences extracted from Spanish OSCAR docs specifying all arguments:
```
python cli.py \
    --dataset oscar \
    --dataset-config unshuffled_deduplicated_es \
    --dataset-split train \
    --text-column text \
    --language es \
    --doc-type sentence \
    --sample 1000 \
    --dimensionality-reduction umap \
    --model-name distiluse-base-multilingual-cased-v1 \
    --output-file perplexity.html
```
# Tests:
```
python -m unittest discover -s ./tests/ -p "test_*.py"
```
