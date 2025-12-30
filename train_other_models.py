import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


@dataclass
class Config:
    data_path: str = "archive/data.csv"

    random_state: int = 42
    test_size: float = 0.20

    use_sample: bool = False
    sample_size: int = 200000

    max_features_tfidf: int = 200000


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_data(cfg: Config) -> tuple[pd.Series, np.ndarray]:
    try:
        df = pd.read_csv(cfg.data_path, encoding="utf-8")
    except Exception:
        passwords: list[str] = []
        strengths: list[int] = []
        with open(cfg.data_path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline()
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                left, sep, right = line.rpartition(",")
                if sep != ",":
                    continue
                pw = left
                try:
                    y = int(right)
                except ValueError:
                    continue
                passwords.append(pw)
                strengths.append(y)
        df = pd.DataFrame({"password": passwords, "strength": strengths})

    if "password" not in df.columns or "strength" not in df.columns:
        # Güvenli parse (virgül içeren satırlar için)
        passwords: list[str] = []
        strengths: list[int] = []
        with open(cfg.data_path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline()
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                left, sep, right = line.rpartition(",")
                if sep != ",":
                    continue
                pw = left
                try:
                    y = int(right)
                except ValueError:
                    continue
                passwords.append(pw)
                strengths.append(y)
        df = pd.DataFrame({"password": passwords, "strength": strengths})

    df = df[["password", "strength"]].dropna()
    df["password"] = df["password"].astype(str)
    df["strength"] = pd.to_numeric(df["strength"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["strength"])
    df["strength"] = df["strength"].astype(int)

    df = df[df["strength"].isin([0, 1, 2])]
    df = df[df["password"].str.len() > 0]

    if cfg.use_sample:
        df = df.sample(n=min(cfg.sample_size, len(df)), random_state=cfg.random_state)

    X_text = df["password"]
    y = df["strength"].to_numpy(dtype=np.int64)
    return X_text, y


def make_splits(cfg: Config, X_text: pd.Series, y: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def compute_manual_features(passwords: pd.Series) -> np.ndarray:
    s = passwords.fillna("").astype(str)

    lengths = s.str.len().astype(np.float32).to_numpy()
    digit_count = s.str.count(r"\d").astype(np.float32).to_numpy()
    upper_count = s.str.count(r"[A-Z]").astype(np.float32).to_numpy()
    lower_count = s.str.count(r"[a-z]").astype(np.float32).to_numpy()
    special_count = s.str.count(r"[^A-Za-z0-9]").astype(np.float32).to_numpy()

    feats = np.stack(
        [
            lengths,
            digit_count,
            upper_count,
            lower_count,
            special_count,
        ],
        axis=1,
    )

    eps = 1e-6
    feats[:, 0] = np.log1p(feats[:, 0] + eps)

    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-6
    feats = (feats - mean) / std

    return feats.astype(np.float32)


def evaluate_and_report(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n[{title}] Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f}\n")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)


def run_logreg_tfidf(cfg: Config) -> None:
    X_text, y = load_data(cfg)
    X_train, X_test, y_train, y_test = make_splits(cfg, X_text, y)

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=cfg.max_features_tfidf,
        dtype=np.float32,
    )

    print("\n[LogReg] TF-IDF fit_transform...")
    X_train_vec = vectorizer.fit_transform(X_train.tolist())
    X_test_vec = vectorizer.transform(X_test.tolist())

    clf = LogisticRegression(
        max_iter=1000,
        random_state=cfg.random_state,
        n_jobs=-1,
        verbose=0,
        multi_class="auto",
    )

    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)

    evaluate_and_report(y_test, y_pred, title="TF-IDF (char 2-5) + LogisticRegression")


def run_rf_manual(cfg: Config) -> None:
    X_text, y = load_data(cfg)
    X_train, X_test, y_train, y_test = make_splits(cfg, X_text, y)

    X_train_man = compute_manual_features(X_train)
    X_test_man = compute_manual_features(X_test)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=cfg.random_state,
        n_jobs=-1,
    )

    clf.fit(X_train_man, y_train)
    y_pred = clf.predict(X_test_man)

    evaluate_and_report(y_test, y_pred, title="RandomForest (manual features)")


def main() -> None:
    cfg = Config()
    set_seed(cfg.random_state)

    run_logreg_tfidf(cfg)
    run_rf_manual(cfg)


if __name__ == "__main__":
    main()
