# import libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

# Veri setinin içeriye aktarılması
df = pd.read_csv("spam.csv", encoding="ISO-8859-1")

# Metin verilerini alalım
documents = df["v2"]
labels = df["v1"] # ham veya spam


# metin temizleme
def clean_text(text):
    
    # Büyük küçük harf çevrimi
    text = text.lower()
    
    # Rakamları temizleme
    text = re.sub(r"\d+","",text)
    
    # Özel karakterlerin kaldırılması
    text = re.sub(r"[^\w\s]","",text)
        
    # Kısa kelimelerin temizlenmesi 
    text  = " ".join([word for word in text.split() if len(word) > 2])
    
    # Stopwordleri at.
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])
    
    return text # Temizlenmiş texti return et

# Metinleri Temizle
cleaned_doc = [clean_text(row) for row in documents]


# %% BOW

# Vectorizer tanimla
vectorizer = CountVectorizer()

# Metin -> Sayısal hale getir
X = vectorizer.fit_transform(cleaned_doc[:75])

# Kelime kümesini göster
feature_names = vectorizer.get_feature_names_out()

# Vektör temsilini göster
vektor_temsili2 = X.toarray()
print(f"Vektör temsili: {vektor_temsili2}")

df_bow = pd.DataFrame(vektor_temsili2, columns = feature_names)

# Kelime frekanslarını göster
word_count = X.sum(axis = 0).A1
word_freg = dict(zip(feature_names,word_count))

# İlk 5 kelimeyi print eettir
most_common_5_words = Counter(word_freg).most_common(5)
print(f"most_common_5_words: {most_common_5_words}")








