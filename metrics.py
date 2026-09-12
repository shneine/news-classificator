from typing import Dict, Iterable, List, Tuple


def accuracy_score(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) == 0:
        return 0.0

    correct = 0
    for real, pred in zip(y_true, y_pred):
        if real == pred:
            correct += 1
    return correct / len(y_true)


def precision_recall_f1_for_class(y_true: List[str], y_pred: List[str], class_name: str) -> Tuple[float, float, float]:
    tp = 0
    fp = 0
    fn = 0

    for real, pred in zip(y_true, y_pred):
        if real == class_name and pred == class_name:
            tp += 1
        elif real != class_name and pred == class_name:
            fp += 1
        elif real == class_name and pred != class_name:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def f1_score_macro(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)
    classes = sorted(set(y_true) | set(y_pred))
    if len(classes) == 0:
        return 0.0

    f1_sum = 0.0
    for class_name in classes:
        _, _, f1 = precision_recall_f1_for_class(y_true, y_pred, class_name)
        f1_sum += f1
    return f1_sum / len(classes)


def classification_report(y_true: Iterable[str], y_pred: Iterable[str]) -> str:
    y_true = list(y_true)
    y_pred = list(y_pred)
    classes = sorted(set(y_true) | set(y_pred))

    lines = []
    lines.append("класс | precision | recall | f1")
    lines.append("-" * 42)
    for class_name in classes:
        precision, recall, f1 = precision_recall_f1_for_class(y_true, y_pred, class_name)
        lines.append(f"{class_name} | {precision:.3f} | {recall:.3f} | {f1:.3f}")
    return "\n".join(lines)


def confusion_matrix(y_true: Iterable[str], y_pred: Iterable[str]) -> Tuple[List[str], List[List[int]]]:
    y_true = list(y_true)
    y_pred = list(y_pred)
    classes = sorted(set(y_true) | set(y_pred))
    index_by_class: Dict[str, int] = {}

    for i, class_name in enumerate(classes):
        index_by_class[class_name] = i

    matrix = []
    for _ in classes:
        matrix.append([0 for _ in classes])

    for real, pred in zip(y_true, y_pred):
        real_index = index_by_class[real]
        pred_index = index_by_class[pred]
        matrix[real_index][pred_index] += 1

    return classes, matrix
