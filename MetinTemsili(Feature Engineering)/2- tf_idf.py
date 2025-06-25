# Import libraries
import pandas as pd 
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

# Belge olustur
documents = [
    "Köpekler sadık hayvanlardır.",
    "Kediler evde beslenebilecek sevimli canlılardır.",
    "Kuşlar gökyüzünde özgürce uçar.",
    "Balıklar suda yaşar ve sessizdir.",
    "Kaplumbağalar yavaş hareket eder.",
    "Aslan ormanın kralı olarak bilinir.",
    "Fil çok büyük ve güçlü bir hayvandır.",
    "Tavşanlar havuç yemeyi sever.",
    "Arılar bal üretir.",
    "İnekler süt verir.",
    "Atlar hızlı koşar.",
    "Kelebekler çiçekleri çok sever.",
    "Maymunlar muz yemeyi çok sever.",
    "Penguenler soğuk iklimlerde yaşar.",
    "Ayılar kış uykusuna yatar.",
    "Tilkiler kurnazlıklarıyla bilinir.",
    "Develer çölde yaşar ve su depolar.",
    "Zürafalar uzun boyunlarıyla dikkat çeker.",
    "Kertenkeleler duvarlarda dolaşabilir.",
    "Ördekler suyu çok sever.",
    "Yılanlar sessizce avlanır.",
    "Kangurular yavrularını keselerinde taşır.",
    "Lemurlar ağaçlarda yaşar.",
    "Kartallar yüksekten uçar ve iyi görür.",
    "Kargalar oldukça zekidir.",
    "Leylekler göç eden kuşlardandır.",
    "Kirpiler kendini tehlikede hissedince top olur.",
    "Baykuşlar gece avlanır.",
    "Serçeler şehirlerde sık görülür.",
    "Martılar deniz kenarında yaşar.",
    "Çitalar dünyanın en hızlı kara hayvanıdır.",
    "Yunuslar insanlar ile iletişim kurabilir.",
    "Kurbağalar su kenarında yaşar.",
    "Horoz sabahları ötmesiyle bilinir.",
    "Tavuklar yumurta verir.",
    "Kazlar sürü halinde gezer.",
    "Koyunlar yünleri için yetiştirilir.",
    "Keçiler dağlık alanlarda rahatça dolaşır.",
    "Domuzlar çamurda oynamayı sever.",
    "Sansarlar gece aktiftir.",
    "Yarasalar ses dalgalarıyla yön bulur.",
    "Timsahlar güçlü çenelere sahiptir.",
    "Karıncalar koloniler halinde çalışır.",
    "Sinekler yaz aylarında çok görülür.",
    "Eşekler yük taşımada kullanılır.",
    "Boğalar saldırgan olabilir.",
    "Örümcekler ağ yaparak avlarını yakalar.",
    "Ahtapotların sekiz kolu vardır.",
    "Midyeler denizlerde yaşar.",
    "Deniz yıldızları kollarıyla hareket eder."
]


# Vektorizer tanımla
tfidf_vectorizer = TfidfVectorizer()

# Metinleri sayisal hale cevir
X = tfidf_vectorizer.fit_transform(documents)

# Kelime kumeleri incele
features_name = tfidf_vectorizer.get_feature_names_out() # get_feature_names_out -> Metinlerimiz içerisindeki her bir kelimenin çantaya aktarılması(Bagofwords)

# Vektor temsilini incele
vektor_temsili = X.toarray()
print(f"tf-idf : {vektor_temsili}") 

df_tfidf = pd.DataFrame(vektor_temsili,columns= features_name)

# Ortalama tf_idf degerlerine bakalim
tf_idf = df_tfidf.mean(axis = 0) # Tf_idf degeri yuksekse metin icerisinde cok fazla gectigini anlariz.