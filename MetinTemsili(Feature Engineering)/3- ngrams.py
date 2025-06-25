# Ä°mport library
from sklearn.feature_extraction.text import CountVectorizer

# ornek metin
documents = [
    "Doğal dil işleme alanı hızla gelişiyor.",
    "Makine öğrenmesi metinleri anlamlandırmak için kullanılır.",
    "NLP modelleri dilin yapısını analiz eder.",
    "Kelime frekansları metin analizinde önemlidir.",
    "TF-IDF yöntemi kelimelere ağırlık verir.",
    "Bag of Words modeli kelime sırasını önemsemez.",
    "Trigram yapısı daha fazla bağlam bilgisi taşır.",
    "Yapay zeka insan dilini anlamaya çalışır.",
    "Chatbotlar kullanıcılarla doğal şekilde konuşur.",
    "Veri ön işleme adımı metin temizliği içerir.",
    "Stopword kelimeler analizden çıkarılır.",
    "Kök bulma işlemi sözcükleri sadeleştirir.",
    "Vektör uzay modeli metinleri sayısal hale getirir.",
    "Embedding yöntemleri anlamsal ilişkileri yakalar.",
    "Kelime benzerliği ölçümleri için vektörler kullanılır.",
    "Transformer tabanlı modeller bağlamı daha iyi anlar.",
    "Dil modeli eğitimi büyük veri gerektirir.",
    "Anlam çıkarımı için bağlam bilgisi şarttır.",
    "Cümleler arasındaki ilişki semantik analizle bulunur.",
    "Derin öğrenme yöntemleri NLP'de yaygınlaşmaktadır."
]



# unigram,bigram, trigram seklinde 3 farkli N degerine sahip gram modeli
vectorizer_unigram = CountVectorizer(ngram_range=(1,1))
vectorizer_bigram = CountVectorizer(ngram_range=(2,2))
vectorizer_trigram = CountVectorizer(ngram_range=(3,3))

# unigram
X_unigram = vectorizer_unigram.fit_transform(documents)
unigram_features = vectorizer_unigram.get_feature_names_out()

# bigram
X_bigram = vectorizer_bigram.fit_transform(documents)
bigram_features = vectorizer_bigram.get_feature_names_out()

# trigram
X_trigram = vectorizer_trigram.fit_transform(documents)
trigram_features = vectorizer_trigram.get_feature_names_out()

# sonuclarin incelenmesi
print(f"unigram_features {unigram_features}")
print(f"bigram_features {bigram_features}")
print(f"trigram_features {trigram_features}")