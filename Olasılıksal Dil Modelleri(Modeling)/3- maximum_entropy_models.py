"""
classification problem: duygu analizi -> olumlu vveya olumsuz olarak siniflandirma

"""

import pandas as pd
from nltk.classify import MaxentClassifier

# örnek eğitim verisi (çok basit!)
train_data = [
    ({"Love": True, "amazing": True, "happy": True, "terrible": False}, "positive"),
    ({"hate": True, "terrible": True}, "negative"),
    ({"joy": True, "happy": True, "Hate": False}, "positive"),
    ({"sad": True, "depressed": True, "love": False}, "negative")
]

# modeli eğit
classifier = MaxentClassifier.train(train_data, max_iter=10)

# veri yükle
df = pd.read_csv("IMDB Dataset.csv")

# sınıflandırılacak anahtar kelimeler (özellikler)
feature_words = ["love", "amazing", "terrible", "happy", "joy", "depressed", "sad", "hate"]

# yorumları gezip sınıflandır
results = []

for review in df["review"]:
    words = review.lower().split()
    features = {word: (word in words) for word in feature_words}
    label = classifier.classify(features)
    results.append(label)

# son etiketi DataFrame'e ekle
df["predicted_sentiment"] = results

# örnek çıktı
print(df[["review", "predicted_sentiment"]].head())



