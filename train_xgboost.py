import os
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- AYARLAR ---
class Config:
    data_path = "data.csv"
    max_features = 2000     # TF-IDF için kelime sayısı (Hız için sınırlı tuttuk)
    ngram_range = (2, 5)    # Karakter n-gram aralığı
    random_state = 42
    test_size = 0.2

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

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

def main():
    set_seed(Config.random_state)
    print(">>> [XGBoost] Veri Yükleniyor...")
    df = load_data(Config.data_path)
    
    X = df["password"]
    y = df["strength"]
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.test_size, random_state=Config.random_state, stratify=y
    )
    
    # TF-IDF Vektörleştirme (Karakter tabanlı)
    print(">>> TF-IDF İşlemi (Karakter tabanlı)...")
    vectorizer = TfidfVectorizer(analyzer="char", 
                                 ngram_range=Config.ngram_range, 
                                 max_features=Config.max_features)
    
    # Bellek sorunu yaşamamak için seyrek matris kullanıyoruz
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # --- XGBoost Modeli ---
    print(">>> XGBoost Modeli Eğitiliyor (Bu işlem GPU olmadan biraz sürebilir)...")
    
    # Eğer Colab'de GPU açıksa 'tree_method': 'gpu_hist' kullanılabilir.
    # Garanti olsun diye standart ayarda bırakıyorum, otomatik algılar.
    model = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6, 
        objective='multi:softprob', 
        num_class=3, 
        random_state=Config.random_state,
        n_jobs=-1  # Tüm işlemcileri kullan
    )
    
    model.fit(X_train_vec, y_train)
    
    print(">>> Tahmin Yapılıyor...")
    y_pred = model.predict(X_test_vec)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n>>> XGBoost Accuracy: {acc:.4f}")
    
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['Zayıf', 'Orta', 'Güçlü']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Confusion Matrix (XGBoost)')
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    plt.show()

if __name__ == "__main__":
    main()