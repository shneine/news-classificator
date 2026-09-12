import argparse
import csv
import os
from collections import Counter
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np

from data_mock import clean_text, get_test_headings, load_headings_from_csv, tokenize
from logistic_regression import predict_log_reg_batch, train_log_reg_multiclass
from metrics import accuracy_score, classification_report, confusion_matrix, f1_score_macro
from naive_bayes import predict_bayes_batch, train_bayes
from svm import predict_svm_batch, train_svm_multiclass
from vectorizer import create_dict, delete_stop_words, vectorize_text, vectorize_texts


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def _get_dataset_path(filename: str) -> str:
    path_in_data = os.path.join(DATA_DIR, filename)
    if os.path.exists(path_in_data):
        return path_in_data
    return os.path.join(BASE_DIR, filename)


DEFAULT_CSV = _get_dataset_path("dataset.csv")

DATASET_FILES = [
    ("dataset.csv", _get_dataset_path("dataset.csv")),
    ("bigger_dataset.csv", _get_dataset_path("bigger_dataset.csv")),
    ("biggest_dataset.csv", _get_dataset_path("biggest_dataset.csv")),
]


def available_sources() -> List[Tuple[str, object]]:
    sources: List[Tuple[str, object]] = [("mock", None)]
    for name, path in DATASET_FILES:
        if os.path.exists(path):
            sources.append((name, path))
    return sources


def load_source(path) -> Tuple[List[str], List[str]]:
    if path is None:
        return get_test_headings()
    return load_headings_from_csv(path)


def collect_datasets(include_mock: bool = True) -> List[Tuple[str, List[str], List[str]]]:
    datasets = []
    for name, path in available_sources():
        if path is None and not include_mock:
            continue
        headings, categories = load_source(path)
        datasets.append((name, headings, categories))
    return datasets


def preprocess_headings(headings: List[str]) -> List[List[str]]:
    tokens_list = []
    for raw_text in headings:
        cleaned = clean_text(raw_text)
        tokens = tokenize(cleaned)
        tokens_without_stop_words = delete_stop_words(tokens)
        tokens_list.append(tokens_without_stop_words)
    return tokens_list


def stratified_train_test_split(
    headings: List[str],
    categories: List[str],
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    if len(headings) != len(categories):
        raise ValueError("Количество заголовков и меток классов должно совпадать.")
    if len(headings) < 2:
        raise ValueError("Для обучения и тестирования нужно минимум 2 заголовка.")
    if not 0 < test_size < 1:
        raise ValueError("test_size должен быть числом от 0 до 1.")

    rng = np.random.default_rng(random_state)
    indices_by_class: Dict[str, List[int]] = {}
    for i, label in enumerate(categories):
        indices_by_class.setdefault(label, []).append(i)

    train_indices = []
    test_indices = []

    for label in sorted(indices_by_class):
        indices = indices_by_class[label]
        indices = list(rng.permutation(indices))

        if len(indices) == 1:
            train_indices.extend(indices)
            continue

        test_count = int(round(len(indices) * test_size))
        test_count = max(1, test_count)
        test_count = min(test_count, len(indices) - 1)

        test_indices.extend(indices[:test_count])
        train_indices.extend(indices[test_count:])

    if len(test_indices) == 0:
        last_index = train_indices.pop()
        test_indices.append(last_index)

    train_indices = list(rng.permutation(train_indices))
    test_indices = list(rng.permutation(test_indices))

    X_train_text = [headings[i] for i in train_indices]
    X_test_text = [headings[i] for i in test_indices]
    y_train = [categories[i] for i in train_indices]
    y_test = [categories[i] for i in test_indices]
    return X_train_text, X_test_text, y_train, y_test


def prepare_vectors(
    train_headings: List[str],
    test_headings: List[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], List[List[str]], List[List[str]]]:
    train_tokens = preprocess_headings(train_headings)
    test_tokens = preprocess_headings(test_headings)

    vocab = create_dict(train_tokens)
    if len(vocab) == 0:
        raise ValueError("Словарь пустой. Проверьте данные после очистки и удаления стоп-слов.")

    X_train = vectorize_texts(train_tokens, vocab)
    X_test = vectorize_texts(test_tokens, vocab)
    return X_train, X_test, vocab, train_tokens, test_tokens


def evaluate_predictions(model_name: str, y_test: List[str], predictions: List[str]) -> Dict[str, object]:
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score_macro(y_test, predictions)
    report = classification_report(y_test, predictions)
    classes, matrix = confusion_matrix(y_test, predictions)

    return {
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "report": report,
        "classes": classes,
        "confusion_matrix": matrix,
    }


def save_metrics_csv(metrics: List[Dict[str, object]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "accuracy", "macro_f1"])
        for row in metrics:
            writer.writerow([row["model"], f"{row['accuracy']:.4f}", f"{row['macro_f1']:.4f}"])


def save_predictions_csv(
    test_headings: List[str],
    y_test: List[str],
    predictions_by_model: Dict[str, List[str]],
    output_path: str,
) -> None:
    model_names = list(predictions_by_model.keys())
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["headline", "true_class"] + model_names)
        for i, heading in enumerate(test_headings):
            row = [heading, y_test[i]]
            for model_name in model_names:
                row.append(predictions_by_model[model_name][i])
            writer.writerow(row)


def print_confusion_matrix(classes: List[str], matrix: List[List[int]]) -> None:
    print("Классы:", ", ".join(classes))
    print("Строки - истинные классы, столбцы - предсказанные классы")
    for class_name, row in zip(classes, matrix):
        print(f"{class_name}: {row}")


def classify_new_headline(raw_heading: str, vocab: Dict[str, int], trained_models: Dict[str, object]) -> Dict[str, str]:
    tokens = preprocess_headings([raw_heading])[0]
    vector = vectorize_text(tokens, vocab)

    predictions = {}
    predictions["Naive Bayes"] = predict_bayes_batch(trained_models["Naive Bayes"], [vector])[0]
    predictions["Logistic Regression"] = predict_log_reg_batch(trained_models["Logistic Regression"], [vector])[0]
    predictions["Linear SVM"] = predict_svm_batch(trained_models["Linear SVM"], [vector])[0]
    return predictions


def start_pipeline(args=None) -> None:
    if args is None:
        args = parse_args()

    print("ЭТАП 1: ЗАГРУЗКА ДАННЫХ")
    if args.use_mock:
        headings, categories = get_test_headings()
        print("Используется временный mock-набор данных из data_mock.py")
    elif os.path.exists(args.csv):
        headings, categories = load_headings_from_csv(
            args.csv,
            text_column=args.text_column,
            label_column=args.label_column,
            delimiter=args.delimiter,
        )
        print(f"Загружен CSV-файл: {args.csv}")
    else:
        print(f"ВНИМАНИЕ: CSV-файл не найден по пути {args.csv}.")
        print("Используется временный mock-набор данных из data_mock.py")
        headings, categories = get_test_headings()

    print(f"Всего заголовков: {len(headings)}")
    print(f"Количество классов: {len(set(categories))}")
    print("Распределение по классам:")
    for label, count in Counter(categories).most_common():
        print(f"  {label}: {count}")

    print("\nЭТАП 2: РАЗДЕЛЕНИЕ НА TRAIN/TEST")
    train_headings, test_headings, y_train, y_test = stratified_train_test_split(
        headings,
        categories,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"Train: {len(train_headings)} объектов")
    print(f"Test: {len(test_headings)} объектов")

    print("\nЭТАП 3: ПРЕДОБРАБОТКА И ВЕКТОРИЗАЦИЯ")
    X_train, X_test, vocab, train_tokens, test_tokens = prepare_vectors(train_headings, test_headings)
    dict_len = len(vocab)
    print(f"Размер словаря: {dict_len}")

    print("\nЭТАП 4: ОБУЧЕНИЕ МОДЕЛЕЙ")
    bayes_model = train_bayes(X_train, y_train, dict_len)
    print("Naive Bayes обучена")

    log_reg_model = train_log_reg_multiclass(
        X_train,
        y_train,
        dict_len,
        epochs=args.log_reg_epochs,
        alpha=args.log_reg_alpha,
    )
    print("Logistic Regression обучена")

    svm_model = train_svm_multiclass(
        X_train,
        y_train,
        dict_len,
        epochs=args.svm_epochs,
        alpha=args.svm_alpha,
        lam=args.svm_lam,
    )
    print("Linear SVM обучена")

    trained_models = {
        "Naive Bayes": bayes_model,
        "Logistic Regression": log_reg_model,
        "Linear SVM": svm_model,
    }

    print("\nЭТАП 5: ТЕСТИРОВАНИЕ И ОЦЕНКА")
    predictions_by_model = {
        "Naive Bayes": predict_bayes_batch(bayes_model, X_test),
        "Logistic Regression": predict_log_reg_batch(log_reg_model, X_test),
        "Linear SVM": predict_svm_batch(svm_model, X_test),
    }

    metrics = []
    for model_name, predictions in predictions_by_model.items():
        model_metrics = evaluate_predictions(model_name, y_test, predictions)
        metrics.append(model_metrics)

        print(f"\nМодель: {model_name}")
        print(f"Accuracy: {model_metrics['accuracy']:.3f}")
        print(f"Macro F1-score: {model_metrics['macro_f1']:.3f}")
        print(model_metrics["report"])
        print_confusion_matrix(model_metrics["classes"], model_metrics["confusion_matrix"])

    print("\nЭТАП 6: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "model_metrics.csv")
    predictions_path = os.path.join(args.output_dir, "test_predictions.csv")
    save_metrics_csv(metrics, metrics_path)
    save_predictions_csv(test_headings, y_test, predictions_by_model, predictions_path)
    print(f"Метрики сохранены: {metrics_path}")
    print(f"Предсказания сохранены: {predictions_path}")

    if args.predict:
        print("\nЭТАП 7: ПРОВЕРКА НОВОГО ЗАГОЛОВКА")
        print(f"Заголовок: {args.predict}")
        new_predictions = classify_new_headline(args.predict, vocab, trained_models)
        for model_name, predicted_class in new_predictions.items():
            print(f"{model_name}: {predicted_class}")


def default_args(csv_path=None, use_mock=False, predict=None) -> SimpleNamespace:
    return SimpleNamespace(
        csv=csv_path or DEFAULT_CSV,
        use_mock=use_mock,
        text_column="headline",
        label_column="category",
        delimiter=",",
        test_size=0.3,
        random_state=42,
        output_dir="results",
        predict=predict,
        log_reg_epochs=250,
        log_reg_alpha=0.05,
        svm_epochs=150,
        svm_alpha=0.01,
        svm_lam=0.01,
    )


def _train_models(headings, categories, test_size=0.3, random_state=42, max_rows=None):
    if max_rows is not None and len(headings) > max_rows:
        idx = np.random.default_rng(random_state).permutation(len(headings))[:max_rows]
        headings = [headings[i] for i in idx]
        categories = [categories[i] for i in idx]
    train_headings, test_headings, y_train, y_test = stratified_train_test_split(
        headings, categories, test_size=test_size, random_state=random_state
    )
    X_train, X_test, vocab, _, _ = prepare_vectors(train_headings, test_headings)
    dict_len = len(vocab)

    models = {
        "Naive Bayes": train_bayes(X_train, y_train, dict_len),
        "Logistic Regression": train_log_reg_multiclass(X_train, y_train, dict_len, epochs=250, alpha=0.05),
        "Linear SVM": train_svm_multiclass(X_train, y_train, dict_len, epochs=150, alpha=0.01, lam=0.01),
    }
    predictions = {
        "Naive Bayes": predict_bayes_batch(models["Naive Bayes"], X_test),
        "Logistic Regression": predict_log_reg_batch(models["Logistic Regression"], X_test),
        "Linear SVM": predict_svm_batch(models["Linear SVM"], X_test),
    }
    metrics = {
        name: (accuracy_score(y_test, pred), f1_score_macro(y_test, pred))
        for name, pred in predictions.items()
    }
    return vocab, models, y_test, predictions, metrics, len(train_headings), len(test_headings)


def _choose_source(sources):
    print("\nДоступные наборы данных:")
    for i, (name, path) in enumerate(sources, 1):
        size = "встроенный набор" if path is None else os.path.basename(str(path))
        print(f"  {i}. {name} ({size})")
    raw = input("Номер набора (Enter — отмена): ").strip()
    if not raw:
        return None
    if not raw.isdigit() or not (1 <= int(raw) <= len(sources)):
        print("Некорректный выбор.")
        return None
    return sources[int(raw) - 1]


def _print_metrics_table(metrics):
    print("\n  Модель                | Accuracy | Macro-F1")
    print("  " + "-" * 44)
    for name, (acc, f1) in metrics.items():
        print(f"  {name:<21} | {acc:>8.3f} | {f1:>8.3f}")


def run_menu():
    state = {"vocab": None, "models": None, "name": None}

    while True:
        sources = available_sources()
        print("\nМЕНЮ:")
        print("  1. Полный анализ набора данных (обучение, оценка, сохранение в results/)")
        print("  2. Классифицировать новый заголовок")
        print("  3. Сравнить модели на всех доступных наборах данных")
        print("  4. Показать распределение классов в наборе")
        print("  0. Выход")
        choice = input("Выберите пункт: ").strip()

        if choice == "0":
            print("Выход.")
            break

        elif choice == "1":
            src = _choose_source(sources)
            if src is None:
                continue
            name, path = src
            print(f"\n>>> Полный конвейёр на наборе «{name}»\n")
            start_pipeline(default_args(csv_path=path, use_mock=(path is None)))

        elif choice == "2":
            if state["models"] is None:
                print("\nСначала нужно обучить модели на одном из наборов.")
                src = _choose_source(sources)
                if src is None:
                    continue
                name, path = src
                headings, categories = load_source(path)
                print(f"Обучение на наборе «{name}»...")
                vocab, models, *_ = _train_models(headings, categories)
                state.update(vocab=vocab, models=models, name=name)
                print("Модели обучены.")
            print(f"\n(используются модели, обученные на наборе «{state['name']}»)")
            while True:
                text = input("\nВведите заголовок (Enter — назад): ").strip()
                if not text:
                    break
                preds = classify_new_headline(text, state["vocab"], state["models"])
                for model_name, predicted in preds.items():
                    print(f"  {model_name}: {predicted}")

        elif choice == "3":
            COMPARE_CAP = 1500
            print("\nСравнение моделей на всех доступных наборах данных...")
            print(f"(для наборов больше {COMPARE_CAP} заголовков обучение идёт "
                  f"на подвыборке — для скорости)")
            datasets = collect_datasets()
            for name, headings, categories in datasets:
                note = ""
                if len(headings) > COMPARE_CAP:
                    note = f" [подвыборка {COMPARE_CAP} из {len(headings)}]"
                *_, metrics, n_train, n_test = _train_models(
                    headings, categories, max_rows=COMPARE_CAP)
                print(f"\nНабор «{name}»: всего {len(headings)}, "
                      f"классов {len(set(categories))}, train {n_train} / test {n_test}{note}")
                _print_metrics_table(metrics)

        elif choice == "4":
            src = _choose_source(sources)
            if src is None:
                continue
            name, path = src
            headings, categories = load_source(path)
            print(f"\nНабор «{name}»: всего заголовков {len(headings)}, "
                  f"классов {len(set(categories))}")
            print("Распределение по классам:")
            for label, count in Counter(categories).most_common():
                print(f"  {label}: {count}")

        else:
            print("Нет такого пункта меню.")


def parse_args():
    parser = argparse.ArgumentParser(description="Классификатор новостных заголовков")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Путь к CSV-корпусу. По умолчанию dataset.csv рядом со скриптом.")
    parser.add_argument("--use-mock", action="store_true", help="Использовать встроенный mock-набор вместо CSV.")
    parser.add_argument("--text-column", default="headline", help="Название колонки с заголовком в CSV (регистр не важен).")
    parser.add_argument("--label-column", default="category", help="Название колонки с классом в CSV (регистр не важен).")
    parser.add_argument("--delimiter", default=",", help="Разделитель CSV-файла.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Доля тестовой выборки.")
    parser.add_argument("--random-state", type=int, default=42, help="Seed для воспроизводимого разбиения.")
    parser.add_argument("--output-dir", default="results", help="Папка для сохранения результатов.")
    parser.add_argument("--menu", action="store_true", help="Запустить интерактивное меню.")
    parser.add_argument("--predict", default=None, help="Новый заголовок для классификации после обучения.")

    parser.add_argument("--log-reg-epochs", type=int, default=250, help="Количество эпох для Logistic Regression.")
    parser.add_argument("--log-reg-alpha", type=float, default=0.05, help="Скорость обучения Logistic Regression.")

    parser.add_argument("--svm-epochs", type=int, default=150, help="Количество эпох для Linear SVM.")
    parser.add_argument("--svm-alpha", type=float, default=0.01, help="Скорость обучения Linear SVM.")
    parser.add_argument("--svm-lam", type=float, default=0.01, help="Коэффициент регуляризации Linear SVM.")
    return parser.parse_args()


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        run_menu()
    else:
        cli_args = parse_args()
        if cli_args.menu:
            run_menu()
        else:
            start_pipeline(cli_args)
