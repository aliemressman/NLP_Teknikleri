# Count vectorizer içeriye aktar.
from sklearn.feature_extraction.text import CountVectorizer

# Veri seti oluştur.
documents = [
    "kedi araba nene teyze",
    "kedi evde"]


# Vectorizer tanımla.
vectorizer = CountVectorizer()


# Metni sayısal vektörlere çevir.
X = vectorizer.fit_transform(documents)

# Kelime kümesi oluşturma [bahçede, evde, kedi]
feature_names = vectorizer.get_feature_names_out()
print(f"kelime kumesi: {feature_names}")

# Vektör temsili
vector_temsili = X.toarray()
print(f"vector_temsili: {vector_temsili}")

"""
kelime kümesi: ['bahçede' 'evde' 'kedi']
vector_temsili: [[1 0 1]
 [0 1 1]]
"""