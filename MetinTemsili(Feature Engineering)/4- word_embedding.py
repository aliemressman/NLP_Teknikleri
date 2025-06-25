"""
word2vec (google)
fasttext (facebook)
"""

# import library
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA # Principle component analysis: dimension reduction (BOYUT İNDİRGEMESİ)

from gensim.models import Word2Vec, FastText
from gensim.utils import simple_preprocess

# ornek veri seti olustur
sentences = [
    # 🐾 Hayvanlar
    "Köpekler sadık hayvanlardır.",
    "Kediler evde yalnız kalmaktan hoşlanmaz.",
    "Papağanlar konuşmayı öğrenebilir.",
    "Atlar hızlı koşabilir.",
    "Balıklar suda yaşar.",
    "Köpek balıkları tehlikeli olabilir.",
    "Kuşlar sabahları öter.",
    "Kedi sessizce yürür.",
    "Köpekler sahiplerini korur.",
    "Köpek kulübesinde uyur.",

    # 🚗 Ulaşım
    "Arabalar yollarda hızla gider.",
    "Otobüs durakta bekler.",
    "Uçaklar havalanmadan önce piste gelir.",
    "Trenler raylarda hareket eder.",
    "Bisiklet çevre dostudur.",
    "Gemi limana yanaştı.",
    "Metro kalabalık olabilir.",
    "Taksi şehir içinde yaygındır.",
    "Uçakla seyahat hızlıdır.",
    "Otobüs yolcularını aldı.",

    # 🍽️ Yemek
    "Elma sağlıklı bir meyvedir.",
    "Pizza çok lezzetliydi.",
    "Çorba sıcak servis edilir.",
    "Kahve sabahları içilir.",
    "Tavuk fırında pişti.",
    "Pilav beyaz pirinçten yapılır.",
    "Karpuz yazın serinletir.",
    "Salata tazeydi.",
    "Bal tatlıdır.",
    "Pasta doğum gününde kesilir."
]


tokenized_sentences = [simple_preprocess(sentece) for sentece in sentences]

# word2vec
word2_vec_model = Word2Vec(sentences= tokenized_sentences,vector_size=50, window=5,min_count=1,sg=0)

# fasttext
fasttext_model = FastText(sentences= tokenized_sentences,vector_size=50, window=5,min_count=1,sg=0)

# gorsellestirme: PCA

def plot_word_embedding(model, title):
    
    word_vectors = model.wv
    
    words = list(word_vectors.index_to_key)[:1000]
    vectors = [word_vectors[word] for word in words]
    
    # PCA
    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(vectors)
    
    # 3d gorsellestirme
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection = "3d")

    # vektorleri ciz
    ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2])
    
    # kelimeleri etiketle
    for i, word in enumerate(words):
        ax.text(reduced_vectors[i, 0], reduced_vectors[i, 1], reduced_vectors[i, 2], word, fontsize=10)


    ax.set_title(title)    
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    plt.show()
    
plot_word_embedding(word2_vec_model, "Word2Vec")
plot_word_embedding(fasttext_model, "Fasttext")


