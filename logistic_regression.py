import numpy as np
from typing import Dict, List, Tuple


BinaryModel = Tuple[np.ndarray, float]
MultiClassModel = Dict[str, object]


def train_log_reg(X_train, Y_train_binary, dict_len: int, epochs: int = 200, alpha: float = 0.05) -> BinaryModel:
    w = np.zeros(dict_len)
    b = 0.0
    X_train = np.array(X_train)
    Y_train_binary = np.array(Y_train_binary)

    for epoch in range(epochs):
        for i in range(len(X_train)):
            x = X_train[i]
            y = Y_train_binary[i]

            z = np.dot(w, x) + b
            z_safe = np.clip(z, -250, 250)
            predict = 1.0 / (1.0 + np.exp(-z_safe))
            error = predict - y

            w = w - (alpha * error) * x
            b = b - alpha * error

    return w, b


def predict_log_reg(w, b: float, new_vector) -> float:
    lin_sum = np.dot(w, new_vector) + b
    lin_sum_safe = np.clip(lin_sum, -250, 250)
    predict = 1.0 / (1.0 + np.exp(-lin_sum_safe))
    return float(predict)


def train_log_reg_multiclass(
    X_train,
    Y,
    dict_len: int,
    epochs: int = 200,
    alpha: float = 0.05,
) -> MultiClassModel:
    X_train = np.array(X_train)
    unique_classes = sorted(set(Y))
    trained_models = {}

    for curr_class in unique_classes:
        binary_Y = []
        for label in Y:
            if label == curr_class:
                binary_Y.append(1)
            else:
                binary_Y.append(0)

        binary_Y = np.array(binary_Y)
        w, b = train_log_reg(X_train, binary_Y, dict_len, epochs=epochs, alpha=alpha)
        trained_models[curr_class] = (w, b)

    return {"classes": unique_classes, "models": trained_models}


def predict_log_reg_multiclass(model: MultiClassModel, test_vector) -> str:
    test_vector = np.array(test_vector)
    class_names = model["classes"]
    trained_models = model["models"]

    prob_list = []
    for curr_class in class_names:
        w, b = trained_models[curr_class]
        prob = predict_log_reg(w, b, test_vector)
        prob_list.append(prob)

    max_index = prob_list.index(max(prob_list))
    return class_names[max_index]


def predict_log_reg_batch(model: MultiClassModel, X) -> List[str]:
    predictions = []
    for vector in X:
        predictions.append(predict_log_reg_multiclass(model, vector))
    return predictions


def classify_multiclass(X_train, Y, dict_len: int, test_vector) -> str:
    model = train_log_reg_multiclass(X_train, Y, dict_len)
    return predict_log_reg_multiclass(model, test_vector)
