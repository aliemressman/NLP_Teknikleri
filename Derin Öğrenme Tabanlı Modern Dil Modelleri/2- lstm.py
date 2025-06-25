"""
metin uretimi
lstm train with text data
text data = gpt ile olustur
"""

# import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# egitim verisi chatgpt ile olustur
texts = [
    "Bugün hava çok güzel, dışarda yürüyüş yapmayı düşünüyorum.",
    "Kitap okumak beni gerçekten mutlu ediyor.",
    "Sabah kahvemi içmeden güne başlayamıyorum.",
    "Film izlemek akşamları beni rahatlatıyor.",
    "Yeni bir hobi edinmek istiyorum, belki resim yapmaya başlarım.",
    "Doğayla vakit geçirmek bana huzur veriyor.",
    "Müzik dinlemek moralimi anında düzeltiyor.",
    "Yarın arkadaşlarımla buluşup kahve içeceğiz.",
    "Son zamanlarda yoga yapmaya başladım ve kendimi daha iyi hissediyorum.",
    "Evde bitki yetiştirmek bana keyif veriyor.",
    "Bugün işe biraz geç kaldım ama yoğunluk yoktu.",
    "Kedim sabah beni erkenden uyandırdı.",
    "Hafta sonu pikniğe gitmeyi planlıyoruz.",
    "Yeni tarifler denemek mutfağı daha eğlenceli hale getiriyor.",
    "Telefonumun şarjı hep en olmadık anda bitiyor.",
    "Spor salonuna yazıldım, motivasyonum yüksek.",
    "Kütüphaneden yeni kitaplar ödünç aldım.",
    "Bazen sadece sessizlik dinlemek istiyorum.",
    "Kahvaltıda menemen yapmayı çok seviyorum.",
    "Bugün kendime zaman ayırmak istiyorum.",
    "Yaz tatili planları yapmaya başladım.",
    "Yeni bir dil öğrenmek bana heyecan veriyor.",
    "Deniz kenarında yürüyüş yapmayı özledim.",
    "Biraz resim çizmek iyi gelebilir.",
    "Dün gece çok güzel bir rüya gördüm.",
    "Sosyal medyada daha az vakit geçirmeye çalışıyorum.",
    "Sabahları meditasyon yapmayı alışkanlık haline getirdim.",
    "Eski dostlarla buluşmak her zaman güzeldir.",
    "Bugün alışverişe çıkmam gerek.",
    "Kendime yeni bir defter aldım, not tutmak için sabırsızlanıyorum.",
    "Ev temizliği yaparken müzik dinlemek daha keyifli.",
    "Bahçede çalışmak bana terapi gibi geliyor.",
    "Kış aylarında battaniyeye sarılıp kitap okumak en sevdiğim şey.",
    "Yeni dizilere başlamak istiyorum ama kararsızım.",
    "Bugün kendimi biraz yorgun hissediyorum.",
    "Sabah yürüyüşleri zihnimi açıyor.",
    "Doğum günü yaklaşıyor, nasıl kutlasam diye düşünüyorum.",
    "Yeni bir proje üzerinde çalışmaya başladım.",
    "Fotoğraf çekmeyi gerçekten çok seviyorum.",
    "Kamp yapmak istiyorum, doğayla baş başa kalmak güzel olurdu.",
    "Bugün bol bol su içmeyi unutmayacağım.",
    "Uzun süredir görüşmediğim arkadaşımı aradım.",
    "Kışlıkları yavaş yavaş kaldırmaya başladım.",
    "Güneşli havalar enerjimi yükseltiyor.",
    "Yeni bir çalma listesi oluşturdum, çok motive edici.",
    "Kahvemi alıp balkonda oturmak harika bir his.",
    "Hafta sonu için küçük bir kaçamak planlıyorum.",
    "Yazmak bana her zaman iyi gelir.",
    "Bugün sadece dinlenmek istiyorum.",
    "Akşam yemeği için ne pişirsem karar veremedim.",
    "İnternetten yeni bir kitap sipariş ettim.",
    "Hava kapalı ama ruh halim oldukça iyi.",
    "Kendi ekmeğimi yapmayı öğrendim, çok keyifli bir süreçti.",
    "Günlük tutmak bana kendimi daha iyi anlatma fırsatı veriyor.",
    "Balkonuma çiçek ektim, açmalarını sabırsızlıkla bekliyorum.",
    "Küçük şeyler de mutlu edebilir insanı.",
    "Bugün telefonumu bir kenara bırakıp doğaya odaklandım.",
    "Sıcak bir çorba bazen en iyi ilaç olabilir.",
    "Anı yaşamak gerektiğini yeniden hatırladım.",
    "Kedimle vakit geçirmek beni gerçekten rahatlatıyor.",
    "İnsan bazen sadece nefes almak ister.",
    "Yeni bir müzik aleti çalmayı öğrenmek istiyorum.",
    "Bazen durup gökyüzüne bakmak gerekir.",
    "Bugün daha çok gülümsedim.",
    "Küçük bir hediye aldım kendime.",
    "Yolda yürürken çocukların oyun sesleri moralimi yükseltti.",
    "Akşam çayı keyfi gibisi yok.",
    "Kendi sınırlarımı fark etmek önemliymiş.",
    "Kardeşimle uzun zamandır bu kadar çok gülmemiştim.",
    "Bugün eski bir fotoğraf albümünü karıştırdım.",
    "Yeni ayakkabılarımı ilk kez giydim.",
    "Bazen hiçbir şey yapmadan oturmak gerekir.",
    "Rüyalar bazen bilinçaltımızla konuşur.",
    "Kış çayı demledim, kokusu evi sardı.",
    "Hayallerim için küçük adımlar atıyorum.",
    "Bugün hayatla barış içindeyim.",
    "Uzun zamandır böyle huzurlu hissetmemiştim.",
    "Dışarıda yağmur yağıyor ama içim sıcacık.",
    "Yeni yıl kararlarımı yeniden gözden geçirdim.",
    "Kütüphanemde okunmamış kitaplar beni bekliyor.",
    "Bugün birine yardım ettim ve çok mutlu oldum.",
    "Kafamı dağıtmak için biraz temizlik yaptım.",
    "Kendi pizzamı yaptım, oldukça başarılıydı.",
    "Dışarda kuş sesleri eşliğinde kahvemi içtim.",
    "Yeni bir podcast keşfettim, çok ilham verici.",
    "Bugün geçmişi değil, geleceği düşünmek istiyorum.",
    "Hayat bazen küçük sürprizlerle güzelleşiyor.",
    "Sabah alarmıma uyanmak zor oldu ama başardım.",
    "Kendi yaptığım takıları takmak hoşuma gidiyor.",
    "Kafamı dinlemek için biraz yürüyüş yaptım.",
    "Bugün kendime nazik davrandım.",
    "Bir dostla kahve içmenin yeri başka.",
    "Pencereden gelen güneş ışığı içimi ısıttı.",
    "Bu sabah erkenden uyanıp gün doğumunu izledim.",
    "Zaman ne çabuk geçiyor, fark etmeden akşam olmuş.",
    "Bugün yeni şeyler öğrenmeye açık bir gündü.",
    "İçsel huzurumu korumaya çalışıyorum.",
    "Etrafıma daha dikkatli bakmaya başladım.",
    "Kendi kararlarımın sorumluluğunu almak istiyorum.",
    "Yaşamın küçük detaylarında mutluluğu buluyorum.",
    "Bugün umutluyum, her şey daha iyi olacak gibi.",
    "Sessiz bir köşe bulup biraz kitap okudum.",
    "Kendime zaman ayırmak hiç bu kadar iyi gelmemişti.",
    "Yeni hedefler belirledim ve motiveyim.",
    "Bugün daha iyi hissediyorum, her şey yoluna girecek."
]



#%% metin temizleme ve prepcrocessing: tokenization, padding, label encoding

# tokenization
tokenizer = Tokenizer() # Kelimeleri sayılara çevirmek için tokenizer nesnesi oluşturulur
tokenizer.fit_on_texts(texts) # Tüm metinlerdeki kelimeleri öğrenir ve indeksler
total_words = len(tokenizer.word_index) + 1 # Toplam kelime sayısı (1 eklenir çünkü index 1'den başlar)

# n-gram dizileri olustur ve padding uygula
input_sequences = [] # N-gram dizileri burada tutulacak
for text in texts:
    # metinleri kelime indekslerine cevir
    token_list = tokenizer.texts_to_sequences([text])[0]

    # her metin icin n-gram dizisini olusturalim
    for i in range(1, len(token_list)): # N-gram'lar oluşturuluyor
        n_gram_sequences = token_list[:i+1] # Örn: [1], [1, 2], [1, 2, 3], ...
        input_sequences.append(n_gram_sequences) # Her birini listeye ekle
         
# Tüm dizileri aynı uzunlukta olacak şekilde başa sıfır ekleyerek hizala (padding)
max_sequence_length = max(len(x) for x in input_sequences)

# dizileri padding islemi uygula, hepsinin ayni uzunlukta olmasini sagla
input_sequences = pad_sequences(input_sequences, maxlen = max_sequence_length, padding = "pre")

# X(girdi) ve y(hedef degisken)
X = input_sequences[:,:-1]
y = input_sequences[:,-1]

y = tf.keras.utils.to_categorical(y,num_classes = total_words) # one hot encoding

#%% LSTM modeli olustur, compile, train ve evaluate

model = Sequential() # Ardışık bir model başlat

# embedding
model.add(Embedding(total_words, 50, input_length = X.shape[1]))
# Kelime indekslerini 50 boyutlu vektörlere dönüştür


# lstm
model.add(LSTM(100,return_sequences = False))
# 100 LSTM birimi ile sıradaki kelimeyi tahmin etmeye çalış (return_sequences=False çünkü son çıktıyı istiyoruz)


# output
model.add(Dense(total_words, activation = "softmax"))
# Çıktıyı tüm kelimelere göre olasılık olarak dağıt

# model compile
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
# Modeli derle (optimizasyon ve kayıp fonksiyonu)

# model training
model.fit(X,y,epochs = 100, verbose = 1)

#%% model prediction

def generate_text(seed_text, next_words):
    for _ in range(next_words): # Belirlenen sayıda kelime üret
        
        # Başlangıç metnini sayılara çevir
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        # Padding uygula
        token_list = pad_sequences([token_list], maxlen= max_sequence_length-1, padding = "pre")

        # prediction
        predicted_probablities = model.predict(token_list, verbose = 0)
        
        # en yuksek olasiliga sahip kelimenin indexini bul
        predicted_word_index = np.argmax(predicted_probablities, axis = -1)
        
        # tokenizer ile kelime indexinden asil kelieyi bul
        predicted_word = tokenizer.index_word[predicted_word_index[0]]
        
        # tahmin edilen kelimeyi seed_text e ekle
        seed_text = seed_text + " " + predicted_word
        
    return seed_text # Tam metni döndür

seed_text = "nasılsın" # Başlangıç metni
print(generate_text(seed_text, 6)) # 6 kelime üret ve yazdır



























