# import libraries

import nltk
from nltk.util import ngrams # n gram modeli oluşturmak için
from nltk.tokenize import word_tokenize # tokenization

from collections import Counter

# ornek veri seti olustur

corpus = {
    "I love apple",
    "I love him",
    "I love NLP",
    "You love me",
    "He loves apple",
    "They love apple",
    "I love you and you love me"}


"""
problem tanima yapalim:
    dil modeli yapmak istiyoruz
    amac 1 kelimeden sonra gelecek kelimeyi tahmin etmek: metin turetmek/olusturmak
    bunun icin n gram dil modelini kullanicaz

    ex: I ...(love) ... (apple)
"""

# verileri token haline getir
tokens = [word_tokenize(sentence.lower()) for sentence in corpus]

#  bigram ile 2 li kelime gruplari olusturalim
bigrams = []
for token_list in tokens:
    bigrams.extend(list(ngrams(token_list, 2)))

bigrams_freg = Counter(bigrams)

# trigram
trigrams = []
for token_list in tokens:
    trigrams.extend(list(ngrams(token_list, 3)))

trigrams_freg = Counter(trigrams)

# model testing

# I love bigramindan sonra "you" veya "apple" kelimelerinin gelme olasiliklarini hesaplayalim
bigram = ("i","love") # Hedef bigram

# "i love you" olma olasiligi
prob_you = trigrams_freg[("i","love","you")]/bigrams_freg[bigram]
print(f"you kelimesinin olma olasiligi: {prob_you}")


# i love apple olma olasiligi
prob_apple = trigrams_freg[("i","love","apple")]/bigrams_freg[bigram]
print(f"apple kelimesinin olma olasiligi: {prob_you}")























