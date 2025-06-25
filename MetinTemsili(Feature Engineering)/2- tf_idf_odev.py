import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
import string 
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")

# Veri setini yükle
df = pd.read_csv("IMDB Dataset.csv", encoding="latin1")

# Yalnızca v2 sütununu al
X = pd.DataFrame(df["review"].astype(str))

# Noktalama işaretlerini kaldır
translator = str.maketrans("","",string.punctuation)
X["review"] = X["review"].apply(lambda x: x.translate(translator))

# Küçük harfe çevir
X["review"] = X["review"].apply(lambda x: x.lower())

# (İsteğe bağlı) Yazım düzeltme — çok yavaş olabilir
# from textblob import TextBlob
# X["v2"] = X["v2"].apply(lambda x: str(TextBlob(x).correct()))

# Stopword temizliği
stop_words = set(stopwords.words("english"))
X["review"] = X["review"].apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))

# TF-IDF
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X["review"])

# Kelimeleri ve skorları al
feature_names = vectorizer.get_feature_names_out()
tfidf_score = X_vect.mean(axis=0).A1

# DataFrame'e dök
df_tfidf = pd.DataFrame({
    "word": feature_names,
    "tfidf_score": tfidf_score
})

# En yüksek skorlu kelimeleri göster
df_tfidf_sorted = df_tfidf.sort_values(by="tfidf_score", ascending=False)
print(df_tfidf_sorted.head(10))
