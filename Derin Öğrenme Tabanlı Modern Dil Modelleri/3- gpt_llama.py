"""
metin uretimi

gpt-2 metin uretimi calismasi
llama
"""

# import libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM # Llama

# modelin tanimlanmasi
model_name = "gpt2"
model_name_llama = "huggyllama/llama-7b"

# tokenizer tanimlama ve model olusturma
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer_llama = AutoTokenizer.from_pretrained(model_name_llama) # llama

model = GPT2LMHeadModel.from_pretrained(model_name)
model_llama = AutoModelForCausalLM.from_pretrained(model_name_llama) # llama

# metin uretimi icin gerekli olan baslangic text i
text = "i go to school for"

# tokenization
inputs = tokenizer.encode(text, return_tensors="pt")
inputs_llama = tokenizer_llama(text, return_tensors = "pt")

# metin uretimi gerceklestirelim
outputs = model.generate(inputs, max_length = 55) # inputs = modelin baslangic noktasi, max_length = max sozcuk sayisi
outputs_llama = model_llama.generate(inputs_llama.inputs_ids, max_length = 55) # llama

# modelin urettigi tokenlari okunabilir hale getirmemiz lazim
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True) # Ozel tokenlari(orn: cumle baslangic ve bitis tokenlari) metinden cikarir
generated_text_llama = tokenizer_llama.decode(outputs[0], skip_special_tokens=True) # llama

# Uretilen metni print ettirelim

print(generated_text)
print(generated_text_llama)







