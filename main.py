import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk

# NLTK Kütüphaneleri
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Makine Öğrenmesi Kütüphaneleri
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. VERİ YÜKLEME ---
print("1. 'spam_veri.csv' Yükleniyor...")

try:
    df = pd.read_csv("spam_veri_seti.csv", encoding="utf-8")
    
    if len(df.columns) >= 2:
        df = df.iloc[:, :2] 
        df.columns = ["message", "label"] 
    
    print(f"   Veri Seti Başarıyla Yüklendi!")
    print(f"   Boyut: {df.shape}")
    print(f"   Sınıf Dağılımı:\n{df['label'].value_counts()}")

except FileNotFoundError:
    print("\nHATA: 'spam_veri_seti.csv' dosyası bulunamadı!")
    print("Lütfen veri setinin adının 'spam_veri_seti.csv' olduğundan ve")
    print("bu kodla (main.py) aynı klasörde olduğundan emin olun.")
    exit()
except Exception as e:
    print(f"\nBeklenmedik bir hata oluştu: {e}")
    exit()

# --- METİN ÖN İŞLEME (NLP) ---
print("\n2. Türkçe Metin Temizliği Yapılıyor...")
turkce_stopwords = set(stopwords.words('turkish'))

def metin_temizle(text):
    # a. Küçük harfe çevir
    text = str(text).lower()
    # b. Noktalama işaretlerini kaldır
    text = "".join([char for char in text if char not in string.punctuation])
    # c. Kelimelere ayır
    words = text.split()
    # d. Etkisiz kelimeleri (ve, veya, ama vb.) temizle
    words = [word for word in words if word not in turkce_stopwords]
    return " ".join(words)

df['clean_message'] = df['message'].apply(metin_temizle)

# --- Sayıya Çevirme ---
print("\n3. Metinler Matematiksel Vektörlere Dönüştürülüyor...")
# En çok kullanılan 3000 kelimeyi özellik olarak seç
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['clean_message']).toarray()
y = df['label']

# Veriyi Test verisi olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MODEL EĞİTİMİ ---
print(f"\n4. Model Eğitiliyor (Eğitim Verisi: {len(X_train)} adet)...")
model = MultinomialNB()
model.fit(X_train, y_train)

# Test verisi ile tahmin yap
y_pred = model.predict(X_test)

# --- SONUÇLAR VE GRAFİK ---
accuracy = accuracy_score(y_test, y_pred)
print(f"\nMODEL BAŞARISI: %{accuracy * 100:.2f}")
print("-" * 30)
print(classification_report(y_test, y_pred))

# Karışıklık Matrisi (Confusion Matrix)
try:
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=['Normal', 'Spam'], 
                yticklabels=['Normal', 'Spam'])
    plt.title('Spam Tespit Performansı')
    plt.ylabel('Gerçek Durum')
    plt.xlabel('Modelin Tahmini')
    plt.show(block=False)
    plt.pause(1) 
except Exception as e:
    print("Grafik çizilirken bir hata oldu (önemsiz), devam ediliyor.")

# --- CANLI TEST SİMÜLASYONU ---
def canli_test():
    print("\n" + "="*50)
    print(" SPAM AVCSISI - CANLI TEST MODU")
    print("="*50)
    print("Çıkmak için 'q' yazıp Enter'a basın.\n")
    
    while True:
        text = input("Mesajı Yazın: ")
        
        if text.lower() == 'q':
            print("Çıkış yapılıyor... Görüşmek üzere!")
            break
        
        # Girilen metni işle ve tahmin et
        cleaned = metin_temizle(text)
        vec = tfidf.transform([cleaned]).toarray()
        
        # Tahmin ve Olasılık
        tahmin = model.predict(vec)[0]
        olasilik = model.predict_proba(vec).max() * 100
        
        if tahmin == 'spam':
            print(f"DİKKAT! Bu mesaj SPAM olabilir. (Güven: %{olasilik:.1f})")
            print("   -> Öneri: Linklere tıklamayın, cevap vermeyin.\n")
        else:
            print(f"GÜVENLİ. Bu mesaj Normal görünüyor. (Güven: %{olasilik:.1f})\n")

canli_test()