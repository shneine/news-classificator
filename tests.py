import os
import random
import tempfile
import unittest

import numpy as np

import main
from data_mock import clean_text, get_test_headings, load_headings_from_csv, tokenize
from logistic_regression import predict_log_reg_multiclass, train_log_reg_multiclass
from metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score_macro,
    precision_recall_f1_for_class,
)
from naive_bayes import predict_bayes, train_bayes
from svm import predict_svm, train_svm_multiclass
from vectorizer import create_dict, delete_stop_words, vectorize_text, vectorize_texts


# Набор данных для интеграционного теста
def get_datasets():
    return main.collect_datasets()


# 1. Предобработка текста
class TestTextPreprocessing(unittest.TestCase):
    def test_clean_text_lowercases_and_strips_punctuation(self):
        self.assertEqual(clean_text("Кто-то, ПРИВЕТ!! Test123"), "кто-то привет test123")

    def test_clean_text_collapses_whitespace(self):
        self.assertEqual(clean_text("много    пробелов\tи\nпереносов"), "много пробелов и переносов")

    def test_tokenize_splits_on_spaces(self):
        self.assertEqual(tokenize("акции банков выросли"), ["акции", "банков", "выросли"])

    def test_delete_stop_words_removes_stopwords_and_single_chars(self):
        tokens = ["в", "мире", "и", "politics", "я"]
        # "в", "и" - стоп-слова; "я" удаляется как односимвольный токен
        self.assertEqual(delete_stop_words(tokens), ["мире", "politics"])


# 2. Векторизация (мешок слов)
class TestVectorizer(unittest.TestCase):
    def test_create_dict_assigns_unique_indices_in_order(self):
        vocab = create_dict([["a", "b"], ["b", "c"]])
        self.assertEqual(vocab, {"a": 0, "b": 1, "c": 2})
        # индексы должны быть уникальными и плотными (0..n-1)
        self.assertEqual(sorted(vocab.values()), list(range(len(vocab))))

    def test_vectorize_text_counts_words_and_ignores_unknown(self):
        vocab = {"a": 0, "b": 1, "c": 2}
        vector = vectorize_text(["a", "a", "c", "неизвестное"], vocab)
        self.assertTrue(np.array_equal(vector, np.array([2.0, 0.0, 1.0])))

    def test_vectorize_texts_returns_matrix(self):
        vocab = {"a": 0, "b": 1}
        matrix = vectorize_texts([["a"], ["a", "b", "b"]], vocab)
        self.assertEqual(matrix.shape, (2, 2))
        self.assertTrue(np.array_equal(matrix, np.array([[1.0, 0.0], [1.0, 2.0]])))


# 3. Метрики
class TestMetrics(unittest.TestCase):
    def test_accuracy_basic(self):
        y_true = ["a", "b", "c", "a"]
        y_pred = ["a", "b", "x", "a"]
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 0.75)

    def test_accuracy_empty_returns_zero(self):
        self.assertEqual(accuracy_score([], []), 0.0)

    def test_precision_recall_f1_for_class(self):
        y_true = ["a", "a", "b", "a"]
        y_pred = ["a", "b", "a", "a"]
        # для класса "a": TP=2, FP=1, FN=1
        precision, recall, f1 = precision_recall_f1_for_class(y_true, y_pred, "a")
        self.assertAlmostEqual(precision, 2 / 3, places=6)
        self.assertAlmostEqual(recall, 2 / 3, places=6)
        self.assertAlmostEqual(f1, 2 / 3, places=6)

    def test_macro_f1_perfect_prediction(self):
        y_true = ["a", "b", "c"]
        self.assertAlmostEqual(f1_score_macro(y_true, y_true), 1.0)

    def test_confusion_matrix_shape_and_counts(self):
        y_true = ["a", "b", "a"]
        y_pred = ["a", "a", "b"]
        classes, matrix = confusion_matrix(y_true, y_pred)
        self.assertEqual(classes, ["a", "b"])
        self.assertEqual(matrix, [[1, 1], [1, 0]])
        # сумма всех ячеек равна числу объектов
        self.assertEqual(sum(sum(row) for row in matrix), len(y_true))

    def test_classification_report_mentions_classes(self):
        report = classification_report(["a", "b"], ["a", "b"])
        self.assertIn("precision", report)
        self.assertIn("a", report)
        self.assertIn("b", report)


# 4. Загрузка CSV (регистронезависимые имена колонок)
class TestCsvLoading(unittest.TestCase):
    def _write_csv(self, header, rows):
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(header + "\n")
            for r in rows:
                f.write(r + "\n")
        self.addCleanup(os.remove, path)
        return path

    def test_capitalized_columns_are_resolved(self):
        # колонки с заглавной буквы, как в реальном корпусе
        path = self._write_csv("Headline,Category", ["Рубль укрепился,Экономика", "ЦСКА выиграл матч,Спорт"])
        headings, categories = load_headings_from_csv(path)  # дефолтные headline/category
        self.assertEqual(headings, ["Рубль укрепился", "ЦСКА выиграл матч"])
        self.assertEqual(categories, ["Экономика", "Спорт"])

    def test_lowercase_columns_also_work(self):
        path = self._write_csv("headline,category", ["Тест,Наука"])
        headings, categories = load_headings_from_csv(path)
        self.assertEqual((headings, categories), (["Тест"], ["Наука"]))

    def test_missing_column_raises(self):
        path = self._write_csv("text,label", ["a,b"])
        with self.assertRaises(ValueError):
            load_headings_from_csv(path)  # колонок headline/category нет

    def test_blank_rows_are_skipped(self):
        path = self._write_csv("Headline,Category", ["Заголовок,Спорт", " , ", "Ещё,Наука"])
        headings, categories = load_headings_from_csv(path)
        self.assertEqual(len(headings), 2)
        self.assertEqual(categories, ["Спорт", "Наука"])


# 5. Обучаемость моделей на простом линейно разделимом примере
#    Два класса с непересекающимися "словами" -> модель обязана их различить.
class TestModelsLearnSeparable(unittest.TestCase):
    def setUp(self):
        # признак 0 встречается только у класса A, признак 1 — только у B
        self.X = [[1, 0], [2, 0], [3, 0], [0, 1], [0, 2], [0, 3]]
        self.Y = ["A", "A", "A", "B", "B", "B"]
        self.dict_len = 2

    def test_naive_bayes_learns(self):
        model = train_bayes(self.X, self.Y, self.dict_len)
        self.assertEqual(predict_bayes(model, [5, 0]), "A")
        self.assertEqual(predict_bayes(model, [0, 5]), "B")

    def test_logistic_regression_learns(self):
        model = train_log_reg_multiclass(self.X, self.Y, self.dict_len, epochs=300, alpha=0.1)
        self.assertEqual(predict_log_reg_multiclass(model, [5, 0]), "A")
        self.assertEqual(predict_log_reg_multiclass(model, [0, 5]), "B")

    def test_svm_learns(self):
        model = train_svm_multiclass(self.X, self.Y, self.dict_len, epochs=300)
        self.assertEqual(predict_svm(model, [5, 0]), "A")
        self.assertEqual(predict_svm(model, [0, 5]), "B")


# 6. Интеграционный тест полного цикла на нескольких наборах данных
class TestPipelineOnDatasets(unittest.TestCase):
    def test_full_cycle_invariants(self):
        datasets = get_datasets()
        self.assertGreaterEqual(len(datasets), 1)

        # Тест проверяет инварианты конвейера, а не качество, поэтому для больших
        # корпусов берём воспроизводимую подвыборку — это делает тест быстрым.
        CAP = 300
        for name, headings, categories in datasets:
            with self.subTest(dataset=name):
                if len(headings) > CAP:
                    rng = random.Random(42)
                    idx = list(range(len(headings)))
                    rng.shuffle(idx)
                    idx = idx[:CAP]
                    headings = [headings[i] for i in idx]
                    categories = [categories[i] for i in idx]
                train_h, test_h, y_train, y_test = main.stratified_train_test_split(
                    headings, categories, test_size=0.3, random_state=42
                )
                X_train, X_test, vocab, _, _ = main.prepare_vectors(train_h, test_h)
                self.assertGreater(len(vocab), 0)

                # для скорости в тесте берём меньше эпох — проверяем инварианты, а не качество
                nb_model = train_bayes(X_train, y_train, len(vocab))
                lr_model = train_log_reg_multiclass(X_train, y_train, len(vocab), epochs=40, alpha=0.05)
                svm_model = train_svm_multiclass(X_train, y_train, len(vocab), epochs=40)

                from logistic_regression import predict_log_reg_batch
                from naive_bayes import predict_bayes_batch
                from svm import predict_svm_batch

                predictions_by_model = {
                    "Naive Bayes": predict_bayes_batch(nb_model, X_test),
                    "Logistic Regression": predict_log_reg_batch(lr_model, X_test),
                    "Linear SVM": predict_svm_batch(svm_model, X_test),
                }

                for model_name, predictions in predictions_by_model.items():
                    metrics = main.evaluate_predictions(model_name, y_test, predictions)

                    # длина предсказаний совпадает с тестовой выборкой
                    self.assertEqual(len(predictions), len(y_test))
                    # метрики в допустимом диапазоне [0, 1]
                    self.assertGreaterEqual(metrics["accuracy"], 0.0)
                    self.assertLessEqual(metrics["accuracy"], 1.0)
                    self.assertGreaterEqual(metrics["macro_f1"], 0.0)
                    self.assertLessEqual(metrics["macro_f1"], 1.0)
                    # матрица ошибок квадратная, сумма ячеек = числу объектов
                    classes = metrics["classes"]
                    matrix = metrics["confusion_matrix"]
                    self.assertEqual(len(matrix), len(classes))
                    for row in matrix:
                        self.assertEqual(len(row), len(classes))
                    self.assertEqual(sum(sum(row) for row in matrix), len(y_test))


if __name__ == "__main__":
    unittest.main(verbosity=2)
