import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords as nltk_stopwords

from languages_id import langs_id


supported_langs_nltk_stopwords = [nltk_id for nltk_id in langs_id["nltk_id"] if nltk_id]
stopwords = {nltk_lang_id: nltk_stopwords.words(nltk_lang_id) for nltk_lang_id in supported_langs_nltk_stopwords}
