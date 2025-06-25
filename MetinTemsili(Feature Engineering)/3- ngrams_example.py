import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Örnek veri kümesi
data = {
    "text": [
        # SPAM
        "hediye kazandınız hemen tıklayın",
        "bu mesaj size özel fırsatlar içeriyor",
        "kazanmak için buraya tıklayın",
        "bedava internet kazanmak için kayıt olun",
        "sınırlı süreli kampanya seni bekliyor",
        "ödül kazandınız hemen iletişime geçin",
        "banka hesabınızla ilgili önemli bilgi",
        "tatil kazandınız şimdi arayın",
        "şimdi kayıt olun ve kazanma şansı yakalayın",
        "hediyeniz hazır almak için tıklayın",
        "mobil ödeme ile hediye kazanın",
        "şüpheli işlem tespit edildi detaylar için tıklayın",
        "senin için büyük fırsat",
        "kart bilgilerinizi güncelleyin",
        "bedava cihaz kazanmak ister misiniz?",
        
        # HAM
        "yarınki toplantı saat onda olacak",
        "ödevini zamanında teslim ettiğin için teşekkür ederim",
        "bugün hava çok güzel",
        "sabah kahvaltıda ne yedin",
        "akşam sinemaya gidelim mi",
        "toplantıdan sonra çay içebiliriz",
        "raporu pazartesiye kadar göndermen gerekiyor",
        "haftaya görüşmek üzere",
        "proje hakkında detayları paylaştım",
        "arkadaşlarınla güzel vakit geçirmeni dilerim",
        "bugün yoğun bir gün geçirdim",
        "sınavdan yüksek not aldım",
        "yarın sabah seni arayacağım",
        "sunum için hazırlık yaptım",
        "doğum günün kutlu olsun"
    ],
    "label": [
        # İlk 15 spam (1)
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,

        # Son 15 ham (0)
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
    ]
}


df = pd.DataFrame(data)

# Eğitim/Test ayır
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.33, random_state=42)

# N-gram ayarları
ngram_configs = {
    "Unigram": (1, 1),
    "Bigram": (2, 2),
    "Trigram": (3, 3)
}

# Sonuçları saklamak için
results = {}

for name, ngram_range in ngram_configs.items():
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

# 🎨 GÖRSELLEŞTİRME
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color=["skyblue", "orange", "lightgreen"])
plt.ylim(0, 1)
plt.title("N-Gram Doğruluk Karşılaştırması")
plt.xlabel("N-Gram Türü")
plt.ylabel("Doğruluk Oranı")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()
