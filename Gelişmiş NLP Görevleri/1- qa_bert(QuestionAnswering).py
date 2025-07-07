from transformers import BertTokenizer, BertForQuestionAnswering
import torch

import warnings
warnings.filterwarnings("ignore")

# squad veri seti üzerinde ince ayar yapilmis bert fiil modeli
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

# bert tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# soru cevaplama gorevi icin ince ayar yapilmis bert modeli
model = BertForQuestionAnswering.from_pretrained(model_name)

# cevaplari tahmin eden fonksiyon
def predict_answer(context, question):
    """
        context = metin
        question = soru
        Amac: metin icerisinden soruyu bulmak
        
        1) tokenize
        2) metnin icerisinden soruyu ara
        3) metnin icerisinden sorunun cevabinin nerelerde olabileceğinin skorlarini return et
        4) sokarlardan tokenlarin indekslerini hesapladik
        5) tokenlari bulduk yani cevabi bulduk
        6) okunabilir olmasi icin tokenlardan string 'e cevirdik
    """
    
    # metni ve soruyu tokenlara ayiralim ve modele uygun hale getirelim
    encoding = tokenizer.encode_plus(question, context, return_tensors = "pt", max_length= 512, truncation = True)
    
    # giris tensorlerini hazirla
    input_ids = encoding["input_ids"] # tokenlarin idleri
    attention_mask = encoding["attention_mask"] # hangi tokenlarin dikkate alinacagini belirtir
    
    # modeli calistir ve skorlari hesapla
    with torch.no_grad():
        start_scores, end_scores = model(input_ids, attention_mask, return_dict = False)

    # en yuksek olasiliga sahip start ve end indekslerini hesapliyor
    start_index = torch.argmax(start_scores, dim=1).item() # baslangic index
    end_index = torch.argmax(end_scores, dim = 1).item() # bitis index
    
    # token id lerini kullanarak cevap metinini elde edelim
    answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][start_index: end_index +1 ])
    
    # tokenlari birlestir ve okunabilir hale getir
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    
    return answer

question = "What is the capital of Turkey"
context = "Ankara, officially the Turkey Republic, is a country whose capital is Ankara"

answer = predict_answer(context, question)
print(f"Answer: {answer}")
