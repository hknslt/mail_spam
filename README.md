# 📧 Türkçe Spam SMS Tespit Sistemi (Turkish Spam SMS Detection)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

Bu proje, Doğal Dil İşleme (NLP) ve Makine Öğrenmesi teknikleri kullanılarak Türkçe SMS mesajlarını **"Normal"** veya **"Spam"** (İstenmeyen) olarak sınıflandıran bir yapay zeka uygulamasıdır. 

Proje, **%97.30** doğruluk oranı ile çalışmakta olup, özellikle Türkiye'deki yaygın spam türleri (bahis, dolandırıcılık, kargo vb.) üzerinde eğitilmiştir.

## 🚀 Proje Özellikleri

* **Veri Seti:** Gerçek ve veri çoğaltma (data augmentation) yöntemleriyle oluşturulmuş 2000+ satırlık dengeli Türkçe veri seti.
* **NLP İşlemleri:** NLTK kütüphanesi ile metin temizliği, stopwords kaldırma ve küçük harf dönüşümü.
* **Vektörleştirme:** TF-IDF (Term Frequency-Inverse Document Frequency) yöntemi.
* **Model:** Scikit-Learn kütüphanesinden **Multinomial Naive Bayes** algoritması.
* **Canlı Test:** Terminal üzerinden anlık mesaj girerek test yapabilme imkanı.

## 📂 Dosya Yapısı

* `main.py`: Projenin ana kaynak kodudur. Veriyi yükler, temizler, modeli eğitir ve canlı test arayüzünü başlatır.
* `spam_veri_seti.csv`: Eğitim ve test için kullanılan veri seti (Message, Label).
* `veri_olustur_tr.py`: (Opsiyonel) Veri setini genişletmek ve sentetik veri üretmek için kullanılan script.

## 🛠️ Kurulum

Projeyi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1.  **Projeyi Klonlayın:**
    ```bash
    git clone [https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git](https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git)
    cd REPO_ADINIZ
    ```

2.  **Gerekli Kütüphaneleri Yükleyin:**
    ```bash
    pip install pandas numpy matplotlib seaborn nltk scikit-learn
    ```

3.  **Projeyi Çalıştırın:**
    ```bash
    python main.py
    ```

## 📊 Model Performansı

Model, test veri seti üzerinde aşağıdaki performansı göstermiştir:

| Metrik | Değer |
| :--- | :--- |
| **Accuracy (Doğruluk)** | **%97.30** |
| **Precision (Spam)** | 0.95 |
| **Recall (Spam)** | 1.00 |
| **F1-Score** | 0.97 |

*Model, test setindeki spam mesajların tamamını (%100 Recall) başarıyla yakalamıştır.*

<img width="602" height="574" alt="Ekran görüntüsü 2025-12-28 140630" src="https://github.com/user-attachments/assets/2cdf7c9a-0e8a-417c-a1a7-f512de18d179" />

## 🖥️ Kullanım Örneği (Canlı Test)

Program çalıştırıldığında terminal ekranında bir giriş alanı açılır:

```text
🕵️‍♂️ SPAM AVCISI - CANLI TEST MODU
==================================================
📩 Mesajı Yazın: Tebrikler iphone kazandınız hemen tıklayın
🔴 DİKKAT! Bu mesaj SPAM olabilir. (Güven: %92.4)

📩 Mesajı Yazın: Yarın akşam sinemaya gidelim mi?
🟢 GÜVENLİ. Bu mesaj Normal görünüyor. (Güven: %88.1)
