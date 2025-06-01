import nltk # Natural Language ToolKit
nltk.download("punkt_tab") # Metni kelime ve cümle bazında tokenlara ayırabilmek için gerekli

text = "Hello, World! How are you? Hello, hi ..."

# Kelime tokenizasyonu: word_tokenize: metni kelimelere ayırır, 
# noktalama işaretleri ve boşluklar ayrı birer token olarak elde edilecektir.
word_tokens = nltk.word_tokenize(text)


# cümle tokenizasyonu: sent_tokenize: Metni cümlelere ayırır. Her bir cümle birer token olur.
sentece_tokens = nltk.sent_tokenize(text)