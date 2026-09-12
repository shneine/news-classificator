import numpy as np
from typing import Dict, List, Tuple


BinaryModel = Tuple[np.ndarray, float]
MultiClassModel = Dict[str, object]


def train_svm(
    X_train,
    Y_train_binary,
    dict_len: int,
    alpha: float = 0.01,
    lam: float = 0.01,
    epochs: int = 100,
) -> BinaryModel:
    w = np.zeros(dict_len)
    b = 0.0
    X_train = np.array(X_train)
    Y_train_binary = np.array(Y_train_binary)

    for epoch in range(epochs):
        for i in range(len(X_train)):
            x = X_train[i]
            y = Y_train_binary[i]
            score = np.dot(w, x) + b
            margin = y * score

            if margin >= 1:
                w = w * (1 - alpha * lam)
            else:
                w = w - alpha * (lam * w - y * x)
                b = b + alpha * y

    return w, b


def train_svm_multiclass(
    X_train,
    Y,
    dict_len: int,
    alpha: float = 0.01,
    lam: float = 0.01,
    epochs: int = 100,
) -> MultiClassModel:
    X_train = np.array(X_train)
    unique_classes = sorted(set(Y))
    trained_models = {}

    for c in unique_classes:
        binary_Y = []
        for label in Y:
            if label == c:
                binary_Y.append(1)
            else:
                binary_Y.append(-1)

        binary_Y = np.array(binary_Y)
        w, b = train_svm(X_train, binary_Y, dict_len, alpha=alpha, lam=lam, epochs=epochs)
        trained_models[c] = (w, b)

    return {"classes": unique_classes, "models": trained_models}


def predict_svm(model: MultiClassModel, test_vector) -> str:
    test_vector = np.array(test_vector)
    classes_list = model["classes"]
    trained_models = model["models"]

    scores_list = []
    for c in classes_list:
        w, b = trained_models[c]
        curr_score = np.dot(w, test_vector) + b
        scores_list.append(curr_score)

    max_index = scores_list.index(max(scores_list))
    return classes_list[max_index]


def predict_svm_batch(model: MultiClassModel, X) -> List[str]:
    predictions = []
    for vector in X:
        predictions.append(predict_svm(model, vector))
    return predictions


def predict_svm_multiclass(X_train, Y, dict_len: int, test_vector) -> str:
    model = train_svm_multiclass(X_train, Y, dict_len)
    return predict_svm(model, test_vector)
