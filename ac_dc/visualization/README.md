# Visualization tool

1) Use get_data_for_visualization.py to get the json gathering examples with their computed statistics for the language you chose.
It uses the streaming mode of the Datasets library, so no need to download the dataset, but you have to download the fasttext model (for the language identification) and the kenlm / sentencepiece models (for the perplexity).

2) Specify the path to this json in visualization.py and run the command "streamlit run visualization.py".
