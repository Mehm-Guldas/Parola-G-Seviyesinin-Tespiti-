import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- AYARLAR ---
class Config:
    data_path = "data.csv"
    sequence_length = 50     # Parola uzunluğu sınırı
    vocab_size = 100         # Maksimum karakter çeşitliliği
    embedding_dim = 32       # Karakter vektör boyutu
    lstm_units = 64          # LSTM nöron sayısı
    epochs = 10              # Eğitim turu (LSTM yavaştır, 5-10 iyidir)
    batch_size = 1024        # Hızlı olsun diye batch büyük tuttum
    random_state = 42

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_data(path):
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except:
        passwords = []
        strengths = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            next(f)
            for line in f:
                line = line.strip()
                if not line: continue
                p_part, sep, s_part = line.rpartition(",")
                if sep:
                    passwords.append(p_part)
                    strengths.append(int(s_part))
        df = pd.DataFrame({"password": passwords, "strength": strengths})
    
    df.dropna(inplace=True)
    df["password"] = df["password"].astype(str)
    df["strength"] = df["strength"].astype(int)
    return df

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def main():
    set_seed(Config.random_state)
    print(">>> [LSTM] Veri Yükleniyor...")
    df = load_data(Config.data_path)
    
    X = df["password"].values
    y = df["strength"].values
    
    print(">>> Tokenization (Harfleri sayıya çevirme)...")
    tokenizer = Tokenizer(char_level=True, lower=True)
    tokenizer.fit_on_texts(X)
    
    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen=Config.sequence_length, padding='post', truncating='post')
    
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Karakter sözlüğü boyutu: {vocab_size}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_pad, y, test_size=0.2, random_state=Config.random_state, stratify=y
    )
    
    # --- LSTM MODEL MİMARİSİ ---
    print(">>> LSTM Modeli Kuruluyor...")
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=Config.embedding_dim, input_length=Config.sequence_length),
        # Bidirectional LSTM: Parolayı hem baştan sona hem sondan başa okur
        Bidirectional(LSTM(Config.lstm_units, return_sequences=False)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    print(">>> Eğitim Başlıyor (Biraz zaman alabilir)...")
    history = model.fit(
        X_train, y_train,
        epochs=Config.epochs,
        batch_size=Config.batch_size,
        validation_split=0.1,
        verbose=1
    )
    
    print("\n>>> Test Değerlendirmesi:")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    plot_history(history)
    
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['Zayıf', 'Orta', 'Güçlü']))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.title('Confusion Matrix (LSTM)')
    plt.show()

if __name__ == "__main__":
    main()