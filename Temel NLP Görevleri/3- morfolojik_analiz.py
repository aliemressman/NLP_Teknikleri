import spacy

nlp = spacy.load("en_core_web_sm")

# incelenecek olan kelime yada kelimeler

word = "I go to school"

# kelimeyi nlp işleminden geçir
doc = nlp(word)

for token in doc:
    print(f"Text:  {token.text}")   # kelimenin kendisi
    print(f"Lemma: {token.lemma_}")      # kelimenin kok hali
    print(f"POS: {token.pos_}")     # kelimenin dil bilgisel özelliği
    print(f"TaG: {token.tag_}")     # kelimenin detaylı dilbilgisel özelliği
    print(f"Depenency:  {token.dep_}") # kelimenin rolü
    print(f"Shape: {token.shape_}") # karakter yapisi
    print(f"Is alpha:  {token.is_alpha}") # kelimenin yalnızca alfabetik karakterden olup olmadigini kontrol eder
    print(f"Is stope: {token.is_stop}") # kelimenin stop words olup olmadigi
    print(f"Morfoloji:  {token.morph}") # kelimenin morfolojik ozellikklerini verir
    print(f"Is plural: {'Number=Plur' in token.morph}") # kelimenin cogul olup olmadigi
    
    print()