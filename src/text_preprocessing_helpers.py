import re

import contractions
from nltk.corpus import stopwords

MAX_TOKEN_LENGTH = 512
stop_words = set(stopwords.words("english"))


def remove_stopwords(text):
    return " ".join(word for word in text.split() if word not in stop_words)


def remove_special_characters(text):
    return re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)


def to_lower(text):
    return text.lower()


def remove_contractions(text):
    return contractions.fix(text)


def truncate_text(text):
    return text[:MAX_TOKEN_LENGTH]


def text_preprocessing_pipeline(text):
    if text.strip() == "":
        return ""
    result = to_lower(text)
    result = remove_contractions(result)
    result = remove_special_characters(result)
    result = remove_stopwords(result)

    return result
