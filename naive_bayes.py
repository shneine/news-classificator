import numpy as np
from typing import Dict, Iterable, List


ModelType = Dict[str, Dict[str, np.ndarray]]


def train_bayes(X, Y, dict_len: int) -> ModelType:
    model = {}
    model["классы"] = {}
    model["слова"] = {}

    X = np.array(X)
    Y = np.array(Y)
    cnt = len(Y)
    unique_classes = sorted(set(Y))

    for c in unique_classes:
        class_cnt = np.sum(Y == c)
        model["классы"][c] = np.log(class_cnt / cnt)

        class_docs = X[Y == c]
        class_word_counts = np.sum(class_docs, axis=0)
        total_words_count = np.sum(class_word_counts)

        numerator = class_word_counts + 1
        denominator = total_words_count + dict_len
        model["слова"][c] = np.log(numerator / denominator)

    return model


def predict_bayes(model: ModelType, new_vector) -> str:
    best_class = ""
    max_score = -float("inf")
    new_vector = np.array(new_vector)

    for c in model["классы"]:
        curr_score = model["классы"][c]
        curr_score += np.dot(new_vector, model["слова"][c])
        if curr_score > max_score:
            max_score = curr_score
            best_class = c

    return best_class


def predict_bayes_batch(model: ModelType, X) -> List[str]:
    predictions = []
    for vector in X:
        predictions.append(predict_bayes(model, vector))
    return predictions
