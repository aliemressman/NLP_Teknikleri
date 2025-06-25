"""
Solve  Classification problem(Sentiment analysis in NLP) with RNN (Deep Learning based Language Model)

duygu analiz -> bir cumkenin etiketlenmesi (positive ve negative)
restaurant yorumlari degerlendirme
"""

# import libraries
import pandas as pd
import numpy as np

from gensim.models import Word2Vec # Metin Temsili

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# create dataset

data =  {
  "text": [
    "Toplantı çok verimli geçti, herkes katkı sağladı.",
    "Yönetici hiç yapıcı değildi, sadece eleştirdi.",
    "Bugünkü ekip çalışması mükemmeldi.",
    "Sürekli fazla mesai yapmak çok yorucu.",
    "Yeni proje tam zamanında yetişti, çok gurur duyuyorum.",
    "İletişim eksikliği yüzünden müşteriyle sorun yaşadık.",
    "Patron geri bildirim verirken çok motive ediciydi.",
    "Toplantı tamamen zaman kaybıydı.",
    "Ekip arkadaşım gerçekten çok destek oldu.",
    "İzin almak için bile defalarca sormam gerekti.",
    "E-posta yanıtları çok hızlı geldi, işler kolaylaştı.",
    "Hiç kimse görev paylaşımına uymuyor.",
    "Yeni ofis ortamı ferah ve motive edici.",
    "Bazı çalışanlar hiç sorumluluk almıyor.",
    "Yönetim ekibinin vizyonu gerçekten ilham verici.",
    "Departmanlar arası iletişim çok zayıf.",
    "Bugün işler planlandığı gibi sorunsuz ilerledi.",
    "Raporları sürekli son dakikada istemeleri stresli.",
    "Eğitim süreci çok öğreticiydi ve keyifli geçti.",
    "Hatalı sistemi düzeltmek günler sürdü.",
    "Yeni işe başlayan arkadaş çok çabuk uyum sağladı.",
    "Yönetici sürekli bağırıyor, ortam çok stresli.",
    "Bugünkü sunum çok başarılı geçti.",
    "Sistem sürekli çöküyor, verimli çalışamıyoruz.",
    "Ekibimiz bu hafta tüm hedeflerini aştı.",
    "Yöneticiler çalışanların fikirlerini hiç dinlemiyor.",
    "Tüm belgeler düzenliydi, işler çok kolay ilerledi.",
    "Ofiste sürekli gürültü var, odaklanmak imkansız.",
    "İş arkadaşlarım sayesinde zorlukları rahatça aştım.",
    "Performans değerlendirmesi adil değildi.",
    "Toplantı öncesi herkes iyi hazırlanmıştı.",
    "Eğitim programı tamamen zaman kaybıydı.",
    "Yöneticim beni takdir etti, çok mutlu oldum.",
    "Yemekhanede bugün yemekler berbattı.",
    "Müşteri toplantısı çok olumlu geçti.",
    "Raporlama sistemi çok karışık ve kullanışsız.",
    "İşyerinde herkes birbirine destek oluyor.",
    "Yeni sistem geçişi tam bir felaketti.",
    "İzin sürecim çok kolay ve sorunsuz ilerledi.",
    "İşe her gün isteksiz geliyorum, hiç keyif almıyorum.",
    "Ofisteki çalışma ortamı çok rahat ve huzurlu.",
    "Yönetici toplantılarda sürekli kesintiye uğratıyor.",
    "Ekip olarak zorlu projeyi başarıyla tamamladık.",
    "İş yükü çok fazla, dengeyi kurmak zor oluyor.",
    "Çalışma arkadaşlarım çok destekleyici ve yardımsever.",
    "Teknolojik altyapı çok eski, işler aksıyor.",
    "Bugünkü eğitim çok faydalı ve ilgi çekiciydi.",
    "İş yerinde adalet eksikliği var, moral bozuk.",
    "Yönetim ekibi çalışanları motive etmeye çalışıyor.",
    "Mesai saatleri çok esnek, aileme daha fazla zaman ayırabiliyorum.",
    "Toplantılar genellikle gereksiz ve uzun sürüyor.",
    "Performans değerlendirmem adil ve objektif yapıldı.",
    "Çalışma arkadaşlarım projeye çok katkı sağladı.",
    "Mola alanları çok kalabalık ve rahatsız edici.",
    "İş yerinde iletişim çok açık ve şeffaf.",
    "Bazı çalışanlar işlerini zamanında teslim etmiyor.",
    "Patronumuz yeniliklere çok açık ve destekleyici.",
    "Yıllık izin taleplerim her seferinde sorun oluyor.",
    "Bugün ekipçe güzel bir başarı elde ettik.",
    "Çalışma koşulları fiziksel olarak çok zorlayıcı.",
    "Yöneticim sürekli olumlu geri bildirim veriyor.",
    "İş arkadaşlarım arasında dedikodu çok fazla.",
    "Çalışma saatlerim esnek, iş ve özel hayat dengede.",
    "Yeni proje için motivasyonumuz çok yüksek.",
    "Ofisteki klima sistemi çalışmıyor, çok sıcak oluyor.",
    "Müşteri geri dönüşleri genellikle olumlu ve memnun edici.",
    "Personel toplantılarında herkes fikrini rahatça paylaşıyor.",
    "İş yerinde stres seviyesi çok yüksek.",
    "Toplantılar genellikle amacına uygun ve verimli geçiyor.",
    "Bazı ekip arkadaşlarım işlerine gerekli özeni göstermiyor.",
    "Bugünkü ekip toplantısı çok yapıcı geçti, herkes katkı sağladı.",
    "Projede yaşanan gecikmeler moralimizi bozdu.",
    "Yönetici son derece destekleyici ve anlayışlıydı.",
    "İş yükü çok fazla, tükenmiş hissediyorum.",
    "Yeni ofis düzeni çalışma verimliliğimizi artırdı.",
    "Toplantılar gereksiz yere çok uzun sürüyor.",
    "Çalışma arkadaşlarım her zaman yardıma hazır.",
    "Performans değerlendirmesi adil yapılmadı.",
    "Eğitim programı çok faydalı ve motive ediciydi.",
    "İş yerinde iletişim kopukluğu var.",
    "Mola zamanlarımız iyi planlanmış ve yeterli.",
    "Yönetici kararlarında sürekli belirsizlik oluyor.",
    "Ekip olarak hedeflerimizi zamanında tamamladık.",
    "Ofisteki hava kalitesi oldukça kötü, rahatsızım.",
    "Patronumuz çalışan fikirlerine değer veriyor.",
    "Mesai saatleri esnek değil, zorlanıyorum.",
    "Projeye olan ilgi ve heyecanım arttı.",
    "İş arkadaşları arasında dedikodu çok yaygın.",
    "Yönetici geri bildirimlerini yapıcı şekilde veriyor.",
    "Çalışma ortamındaki gürültü konsantrasyonumu etkiliyor."
  ],
  "label": [
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "positive",
    "negative",
    "positive",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "positive",
    "negative",
    "positive",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative"
  ]
}

df = pd.DataFrame(data)

# %% metin temizleme ve preprcessing : tokenization, padding, label encoding, train test split

# tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])
word_index = tokenizer.word_index

# padding process
maxlen = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen= maxlen)
print(X.shape)

# label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# %% metin temsili: word embedding: word2vec

sentences = [text.split() for text in df["text"]]
word2vec_model = Word2Vec(sentences, vector_size=100,window=5,min_count=1) # UYARIII

embedding_dim = 100
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]


# %% modelling : build, train v e test rnn modeli 

# build model
model = Sequential()

# embedding
model.add(Embedding(input_dim=len(word_index) + 1,
                    output_dim=embedding_dim,
                    weights=[embedding_matrix],
                    input_length=maxlen,
                    trainable=False))

# RNN layer
model.add(SimpleRNN(50, return_sequences=False))

# output layer
model.add(Dense(1, activation="sigmoid"))

# compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# train model
model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test))


# evaluate rnn model

test_loss, test_accuracy = model.evaluate(X_test,y_test)
print(f"Test loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# cumle siniflandirma calismasi 
def classify_sentence(sentence):
    
    seq = tokenizer.texts_to_sequences([sentence])
    padded_seq = pad_sequences(seq,maxlen = maxlen)
    
    prediction = model.predict(padded_seq)
    
    predicted_class = (prediction > 0.5).astype(int)
    label = "positive" if predicted_class[0][0] == 1 else "negative"
    
    return label


sentence = "İş yeri beklediğimden çok daha kötümsü gibiydi ama iyi çıktı."

result = classify_sentence(sentence)
print(f"Result: {result}")



