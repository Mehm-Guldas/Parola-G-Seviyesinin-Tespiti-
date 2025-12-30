import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf


@dataclass
class Config:
    data_path: str = "archive/data.csv"

    random_state: int = 42
    test_size: float = 0.20
    val_size_from_train: float = 0.125  # 0.125 of 0.8 => 0.10 overall

    batch_size: int = 1024
    epochs: int = 10

    max_tokens: int = 50000
    ngrams_up_to: int = 5

    use_sample: bool = False
    sample_size: int = 200000

    run_sklearn_baseline: bool = True
    sklearn_max_features: int = 200000


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compute_manual_features(passwords: pd.Series) -> np.ndarray:
    s = passwords.fillna("").astype(str)

    lengths = s.str.len().astype(np.float32).to_numpy()
    digit_count = s.str.count(r"\d").astype(np.float32).to_numpy()
    upper_count = s.str.count(r"[A-Z]").astype(np.float32).to_numpy()
    lower_count = s.str.count(r"[a-z]").astype(np.float32).to_numpy()
    special_count = s.str.count(r"[^A-Za-z0-9]").astype(np.float32).to_numpy()

    unique_count = s.apply(lambda x: len(set(x))).astype(np.float32).to_numpy()

    has_digit = (digit_count > 0).astype(np.float32)
    has_upper = (upper_count > 0).astype(np.float32)
    has_lower = (lower_count > 0).astype(np.float32)
    has_special = (special_count > 0).astype(np.float32)

    feats = np.stack(
        [
            lengths,
            digit_count,
            upper_count,
            lower_count,
            special_count,
            unique_count,
            has_digit,
            has_upper,
            has_lower,
            has_special,
        ],
        axis=1,
    )

    eps = 1e-6
    feats[:, 0] = np.log1p(feats[:, 0] + eps)
    feats[:, 5] = np.log1p(feats[:, 5] + eps)

    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-6
    feats = (feats - mean) / std

    return feats.astype(np.float32)


def load_data(cfg: Config) -> tuple[pd.Series, np.ndarray]:
    try:
        df = pd.read_csv(cfg.data_path)
        df = df[["password", "strength"]].dropna()
    except Exception:
        passwords: list[str] = []
        strengths: list[int] = []

        with open(cfg.data_path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline()
            if not header.lower().strip().startswith("password"):
                raise ValueError("Beklenmeyen CSV formatı: header bulunamadı")

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
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_train_text,
        y_train,
        test_size=cfg.val_size_from_train,
        random_state=cfg.random_state,
        stratify=y_train,
    )

    return (X_train_text, y_train), (X_val_text, y_val), (X_test_text, y_test)


def build_vectorizer(cfg: Config) -> tf.keras.layers.TextVectorization:
    return tf.keras.layers.TextVectorization(
        standardize=None,
        split="character",
        output_mode="tf-idf",
        max_tokens=cfg.max_tokens,
        ngrams=cfg.ngrams_up_to,
    )


def build_model(
    cfg: Config,
    num_manual_features: int,
    vectorizer: tf.keras.layers.TextVectorization,
) -> tf.keras.Model:
    text_in = tf.keras.Input(shape=(1,), dtype=tf.string, name="password")
    manual_in = tf.keras.Input(shape=(num_manual_features,), dtype=tf.float32, name="manual")
    x_text = vectorizer(text_in)

    x = tf.keras.layers.Concatenate()([x_text, manual_in])
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    out = tf.keras.layers.Dense(3, activation="softmax")(x)

    model = tf.keras.Model(inputs={"password": text_in, "manual": manual_in}, outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.vectorizer = vectorizer
    return model


def make_tf_dataset(texts: pd.Series, manual_feats: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool):
    ds = tf.data.Dataset.from_tensor_slices(({"password": texts.to_numpy(), "manual": manual_feats}, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(200000, len(texts)), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def plot_history(history: tf.keras.callbacks.History) -> None:
    hist = history.history
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(hist.get("loss", []), label="train")
    plt.plot(hist.get("val_loss", []), label="val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist.get("accuracy", []), label="train")
    plt.plot(hist.get("val_accuracy", []), label="val")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def evaluate_and_report(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n[{title}] Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f}\n")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def show_misclassified_examples(texts: pd.Series, y_true: np.ndarray, y_pred: np.ndarray, n: int = 25) -> None:
    idx = np.where(y_true != y_pred)[0]
    if len(idx) == 0:
        print("Yanlış sınıflanan örnek yok (bu beklenmeyebilir, örneklem/küçük veri olabilir).")
        return

    rng = np.random.default_rng(42)
    pick = rng.choice(idx, size=min(n, len(idx)), replace=False)

    print("\n--- Yanlış Sınıflanan Örnekler (rastgele) ---")
    for i in pick:
        print(f"password={texts.iloc[i]!r} | true={y_true[i]} | pred={y_pred[i]}")


def run_keras_mlp(cfg: Config) -> None:
    X_text, y = load_data(cfg)
    (X_train_text, y_train), (X_val_text, y_val), (X_test_text, y_test) = make_splits(cfg, X_text, y)

    print("Split sizes:")
    print("train:", len(X_train_text), "val:", len(X_val_text), "test:", len(X_test_text))

    print("\nClass distribution:")
    for name, yy in [("train", y_train), ("val", y_val), ("test", y_test)]:
        unique, counts = np.unique(yy, return_counts=True)
        print(name, dict(zip(unique.tolist(), counts.tolist())))

    X_train_manual = compute_manual_features(X_train_text)
    X_val_manual = compute_manual_features(X_val_text)
    X_test_manual = compute_manual_features(X_test_text)

    print("\nAdapting TextVectorization (TF-IDF) on training data...")
    vectorizer = build_vectorizer(cfg)
    vectorizer.adapt(X_train_text.to_numpy())

    model = build_model(cfg, num_manual_features=X_train_manual.shape[1], vectorizer=vectorizer)

    train_ds = make_tf_dataset(X_train_text, X_train_manual, y_train, cfg.batch_size, shuffle=True)
    val_ds = make_tf_dataset(X_val_text, X_val_manual, y_val, cfg.batch_size, shuffle=False)
    test_ds = make_tf_dataset(X_test_text, X_test_manual, y_test, cfg.batch_size, shuffle=False)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    plot_history(history)

    proba = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(proba, axis=1)

    evaluate_and_report(y_test, y_pred, title="Keras MLP (Char TF-IDF + manual feats)")
    show_misclassified_examples(X_test_text.reset_index(drop=True), y_test, y_pred, n=25)


def run_sklearn_baseline(cfg: Config) -> None:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
    except Exception as e:
        print("scikit-learn baseline için import hatası:", e)
        return

    X_text, y = load_data(cfg)
    (X_train_text, y_train), (_, _), (X_test_text, y_test) = make_splits(cfg, X_text, y)

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=cfg.sklearn_max_features,
        dtype=np.float32,
    )

    print("\n[sklearn] TF-IDF fit_transform...")
    X_train = vectorizer.fit_transform(X_train_text.tolist())
    X_test = vectorizer.transform(X_test_text.tolist())

    clf = LinearSVC()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    evaluate_and_report(y_test, y_pred, title="TF-IDF (char 2-5) + LinearSVC")


def main() -> None:
    cfg = Config()
    set_seed(cfg.random_state)

    run_keras_mlp(cfg)

    if cfg.run_sklearn_baseline:
        run_sklearn_baseline(cfg)


if __name__ == "__main__":
    main()
