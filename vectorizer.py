import numpy as np
from typing import Dict, Iterable, List


STOP_WORDS = {
    "в", "во", "на", "под", "и", "а", "но", "с", "со", "по", "к", "ко", "из", "у",
    "о", "об", "от", "до", "за", "для", "над", "при", "без", "через", "после",
    "это", "как", "что", "или", "не", "же", "ли", "бы", "его", "ее", "их",
}


def delete_stop_words(tokens: Iterable[str]) -> List[str]:
    clean_tokens = []
    for token in tokens:
        if token not in STOP_WORDS and len(token) > 1:
            clean_tokens.append(token)
    return clean_tokens


def create_dict(clean_texts_array: Iterable[Iterable[str]]) -> Dict[str, int]:
    vocab = {}
    curr_ind = 0
    for text in clean_texts_array:
        for word in text:
            if word not in vocab:
                vocab[word] = curr_ind
                curr_ind += 1
    return vocab


def vectorize_text(cleaned_text: Iterable[str], vocab: Dict[str, int]) -> np.ndarray:
    dict_len = len(vocab)
    vector = np.zeros(dict_len)
    for word in cleaned_text:
        if word in vocab:
            ind = vocab[word]
            vector[ind] += 1
    return vector


def vectorize_texts(cleaned_texts: Iterable[Iterable[str]], vocab: Dict[str, int]) -> np.ndarray:
    vectors = []
    for text in cleaned_texts:
        vectors.append(vectorize_text(text, vocab))
    return np.array(vectors)
