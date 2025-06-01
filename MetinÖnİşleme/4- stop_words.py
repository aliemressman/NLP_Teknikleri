import nltk
from nltk.corpus import stopwords
nltk.download("stopwords") # Farklı dillerde en çok kullanılan stop words içeren veri seti



# İngilizce stop words analizi (nltk)
stop_words_eng = set(stopwords.words("english"))

# Örnek ingilizce metin

text = "There are some examples of handling stop words from some texts."
text_list = text.split()
# Eğer word ingilizce stop wrods listesinde yoksa (stop_words_eng) yoksa, 
# bu kelimeyi filtrelenmiş kisteye ekliyoruz.
filtered_words = [word for word in text_list if word.lower() not in stop_words_eng]
print(f"filtered_words: {filtered_words}")


# %% Türkçe stop words analizi (nltk)
stop_words_tr = set(stopwords.words("turkish"))

# Örnek türkçe metin

metin = "merhaba arkadaslar çok güzel bir ders işliyoruz. Bu ders faydalı mı"
metin_list = metin.split()

filtered_words_tr = [word for word in metin_list if word.lower() not in stop_words_tr]

filtered_cumle = " ".join(filtered_words_tr)
# %% Kütüphanesiz stop words çıkarımı

# Stop Words listesi oluştur.
tr_stopwords = ["için","bu","ile","mu","mi","özel"]

# Örnek Türkçe metin
metin = "Bu bir denemedir. Amacımiz bu metinde bulunan özel karakterleri elemek mi acaba?"

filtered_words = [word for word in metin.split() if word.lower() not in tr_stopwords]
filtered_stopwords = set([word.lower() for word in metin.split() if word.lower() in tr_stopwords])

print(f"filtered_words: {filtered_words}")
