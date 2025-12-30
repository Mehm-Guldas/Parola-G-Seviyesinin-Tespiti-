# Parola Güç Seviyesinin Tespiti - Password Strength Detection

Yapay Sinir Ağları ve Doğal Dil İşleme Teknikleri ile Parola Güç Seviyesinin Tespiti ve Sınıflandırılması

## 📋 Proje Özeti

Bu proje, makine öğrenmesi ve derin öğrenme tekniklerini kullanarak parolaların güç seviyesini (Zayıf, Orta, Güçlü) otomatik olarak tespit eden akıllı modeller geliştirmeyi amaçlamaktadır.

### 🎯 Amaç
- Siber güvenlik alanında parola güvenliğini artırmak
- Kullanıcıları zayıf parolalar konusunda uyarmak
- Farklı makine öğrenmesi algoritmalarının performansını karşılaştırmak

## 📊 Veri Seti

Proje, Kaggle'dan alınan **"Password Strength Classifier Dataset"** kullanılmaktadır:
- **Toplam Veri:** ~670.000 parola
- **Sınıflar:** 
  - 0: Zayıf (Weak)
  - 1: Orta (Medium) 
  - 2: Güçlü (Strong)
- **Kaynak:** [Jikadara, B. (2025)](https://www.kaggle.com/datasets/bhavikbb/password-strength-classifier-dataset/data)

## 🛠️ Kullanılan Teknolojiler

### Programlama Dilleri ve Kütüphaneler
- **Python** - Ana programlama dili
- **TensorFlow/Keras** - Derin öğrenme modelleri
- **XGBoost** - Gradyan boosting algoritması
- **Scikit-learn** - Makine öğrenmesi araçları
- **Pandas & NumPy** - Veri işleme
- **Matplotlib & Seaborn** - Görselleştirme

### Uygulanan Modeller
1. **LSTM (Long Short-Term Memory)** - Derin öğrenme yaklaşımı
2. **XGBoost** - Gradyan boosting
3. **Diğer Makine Öğrenmesi Modelleri**

## 📁 Proje Yapısı

```
parola_guc/
├── train_char_cnn.py              # LSTM modeli eğitimi
├── train_xgboost.py               # XGBoost modeli eğitimi  
├── train_other_models.py          # Diğer ML modelleri
├── train_colab_password_strength.py # Google Colab için optimize edilmiş
├── archive/
│   └── data.csv                   # Eğitim veri seti
├── requirements.txt               # Python bağımlılıkları
├── .gitignore                     # Git ignore dosyası
└── README.md                      # Proje dokümantasyonu
```

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler
```bash
pip install -r requirements.txt
```

### Modelleri Çalıştırma

#### LSTM Modeli
```bash
python train_char_cnn.py
```

#### XGBoost Modeli
```bash
python train_xgboost.py
```

#### Diğer ML Modelleri
```bash
python train_other_models.py
```

## 📈 Özellikler ve Metotlar

### Veri Ön İşleme
- **TF-IDF (Term Frequency-Inverse Document Frequency)** - Karakter tabanlı vektörleştirme
- **Tokenization** - Karakter seviyesinde sayısal dönüşüm
- **Feature Engineering** - Parola uzunluğu, karakter çeşitliliği

### Model Özellikleri
- **LSTM:** Bidirectional LSTM mimarisi ile karakter dizilerini analiz eder
- **XGBoost:** Karakter n-gram'ları ile yüksek performanslı sınıflandırma
- **Optimizasyon:** Hızlı eğitim için batch processing ve GPU desteği

## 📊 Sonuçlar ve Performans

Modeller aşağıdaki metriklerle değerlendirilir:
- **Accuracy** - Doğruluk oranı
- **Classification Report** - Precision, Recall, F1-Score
- **Confusion Matrix** - Karışıklık matrisi
- **Training History** - Eğitim grafikleri

## 🔬 Bilimsel Katkı

### Kaynakça
- Melicher, W., et al. (2016). Fast, lean, and accurate: Modeling password guessability using neural networks. USENIX Security 16.
- Rehman, H., et al. (2024). Password Strength Classification Using Machine Learning Methods. GCWOT 2024.

### Yenilikçi Yaklaşımlar
- Karakter seviyesinde derin öğrenme
- Çoklu model karşılaştırması
- Optimize edilmiş TF-IDF özellik çıkarımı
<img width="1150" height="490" alt="Ekran görüntüsü 2025-12-25 160621" src="https://github.com/user-attachments/assets/20bb4545-7a2b-4001-85b8-d612ece916dc" />
<img width="645" height="372" alt="Ekran görüntüsü 2025-12-25 160633" src="https://github.com/user-attachments/assets/b894b56e-eb05-405d-b044-1336fad30325" />
<img width="769" height="493" alt="Ekran görüntüsü 2025-12-25 164919" src="https://github.com/user-attachments/assets/ee7e2365-39a2-481a-915b-fe08541e5b86" />

## 👨‍💻 Yazar

**Mehmet Şirin Güldaş**  
Bilgisayar Mühendisliği, Trakya Üniversitesi  
Danışman: Dr. Öğr. Üyesi Turgut DOĞAN

## 📄 Lisans

Bu proje akademik çalışma olup araştırma amaçlı kullanılmıştır.

## 🔗 Bağlantılar

- **GitHub:** https://github.com/Mehm-Guldas/Parola-G-Seviyesinin-Tespiti-
- **Dataset:** [Kaggle Password Strength Classifier](https://www.kaggle.com/datasets/bhavikbb/password-strength-classifier-dataset/data)

---

**Anahtar Kelimeler:** Siber Güvenlik, Parola Gücü, Yapay Sinir Ağları, LSTM, XGBoost, Sınıflandırma, TF-IDF, Makine Öğrenmesi
