# import library
from transformers import BertTokenizer, BertModel

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# tokenizer and model create
model_name = "bert-base-uncased" # kucuk boyutlu bert modeli
tokenizer = BertTokenizer.from_pretrained(model_name) # tokenizer yukle
model = BertModel.from_pretrained(model_name) # onceden egitilmis bert modeli


# veri olustur: karsilastirilacak belgeleri ve sorgu cumlemizi olustur

documents = [
    "Machine learning is a field of artificial intelligence",
    "Natural language processing involves understanding human language",
    "Artificial intelligence encomppases machine learning and natural language processing (nlp)",
    "Deep learning is a subset of machine learning",
    "Data science combines statistics, adta analysis and machine learnng",
    "I go to shop"
    ]

query = "What is machine learning?"

# bert ile bilgi getirimi

def get_embedding(text):
    
    # tokenization
    inputs = tokenizer(text, return_tensors="pt", truncation = True, padding = True)
    
    # modeli calistir
    outputs = model(**inputs)
    
    # son gizli katmani alalim
    last_hidden_state = outputs.last_hidden_state
    
    # metni temsili olustur
    embedding = last_hidden_state.mean(dim=1)
    
    # vektoru numpy olarak return et
    return embedding.detach().numpy()

# belgeler ve sorgu icin embedding vektorlerini al
doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])
query_embedding = get_embedding(query)

# kosinus benzerligi ile belgeler arasinda benzerligi hesaplayalim
similarities = cosine_similarity(query_embedding, doc_embeddings)

# her belgenin benzerlik skoıru
for i, score in enumerate(similarities[0]):
    print(f"Documents: {documents[i]}\ n{score}")






