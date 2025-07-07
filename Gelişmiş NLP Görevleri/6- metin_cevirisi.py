from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-fr-en" # en to fr
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = "Bonjour, quel est votre nom"

# encode edelim, sonrasinda modele inout olarak verelim

translated_text = model.generate(**tokenizer(text, return_tensors = "pt", padding = True))

# translated text metne donusturulur
translated_text = tokenizer.decode(translated_text[0], skip_special_tokens=True)
print(f"Translated_text: {translated_text}")