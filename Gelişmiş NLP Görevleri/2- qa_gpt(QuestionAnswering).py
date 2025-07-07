from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

import warnings
warnings.filterwarnings("ignore")

model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_answer(context, question):
    
    input_text = f"Question: {question}, Context: {context}. Please answer the question according to context."
    
    # tokenize
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    
    # modeli calistir
    with torch.no_grad():
        outputs = model.generate(inputs, max_length = 500)
        
    # uretilen yaniti decode edelim
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True) # merhaba <EOS><PAD>
    
    # yanitlari ayikla
    answer = answer.split("Anwser:")[-1].strip()
    
    return answer

question = "Bu hikâyenin başkahramanı kimdir? Neden?"
context = "Ali sabah erkenden kalktı, annesiyle vedalaştı ve büyük bir heyecanla üniversiteye gitmek için yola çıktı. Yeni bir şehirde, yeni bir hayat onu bekliyordu."

answer = generate_answer(context, question)
print(f"Answer: {answer}")