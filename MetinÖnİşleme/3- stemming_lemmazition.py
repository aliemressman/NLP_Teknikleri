import nltk
nltk.download("wordnet") # wordnet: Lemmatization işlemi için gerekli veri tabanı
from nltk.stem import PorterStemmer # Stemming için fonksiyon

# Porter Stemmer nesnesini oluştur.
stemmer = PorterStemmer()

words = ["running","runner","ran","runs","better","go","went"]

# Kelimelerin Stemlerini buluyoruz, bunu yaparkende porter Stemmerın stem() fonksiyonunu kullanıyoruz.
stems = [stemmer.stem(w) for w in words]
print(f"Stem: {stems}")

# %% Lemmaziation

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
words = ["running","runner","ran","runs","better","go","went"]
lemmas = [lemmatizer.lemmatize(w ,pos = "v") for w in words]
print(f"Lemmas: {lemmas}")