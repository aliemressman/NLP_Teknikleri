import string 
from nltk.tokenize import word_tokenize
# import nltk
# nltk.download("wordnet") # wordnet: Lemmatization işlemi için gerekli veri tabanı
# from nltk.stem import PorterStemmer
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

cumle = "NLTK ve spaCy gibi araçlar, doğal dil işleme projelerinde metinleri analiz etmek için oldukça faydalıdır."

# küçük harf çevir
cumle = cumle.lower()

# Noktalamaları kaldır.
cumle = cumle.translate(str.maketrans("","",string.punctuation))

# cümle :"nltk ve spacy gibi araçlar doğal dil işleme projelerinde metinleri analiz etmek için oldukça faydalıdır" 
# Tokenize Et
cumle_tokenized = word_tokenize(cumle)

# Stemlerine ayır. GEREK YOK.
# stemmer = PorterStemmer()
# cumle_stem = [stemmer.stem(word) for word in cumle_tokenized] 

# StopWords temizle.
stop_words = set(stopwords.words("turkish"))

filtreli_cumle = [word for word in cumle_tokenized if word not in stop_words]
filtreli_cumle = " ".join(filtreli_cumle)
print(f"Yeni Cümle: {filtreli_cumle}\nEski Cümle: {cumle}")