"""
varlik ismi tanima: metin(cumle) -> metin icerisinde bulunan varlik isimlerini tanimla 
"""

# import libraries
import pandas as pd
import spacy.cli
spacy.cli.download("en_core_web_sm")
# spacy modeli ile varlik ismi tanimla 
nlp = spacy.load("en_core_web_sm") # spacy kutuphanesi ingilizce dil modeli

content = "Alice works at Amazon and lives in London. She visited British Museum last weekend."

doc = nlp(content) # Bu islem metindeki varliklari analiz eder


for ent in doc.ents:
    # ent.text : varlik ismi(Amazon, Alice)
    # ent.start_char ve ent.end_char :varligin metindeki baslangic ve bitis karakterleri
    # ent.label_ : varligin turu
    print(ent.text, ent.label_)

# ent.lemma_: varligin kok hali
entities = [(ent.text, ent.label_, ent.lemma_) for ent in doc.ents]

# varlik listesini pandas df cevir

df = pd.DataFrame(entities,columns = ["text", "type","lemma"])