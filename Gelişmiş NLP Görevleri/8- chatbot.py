import openai

openai.api_key="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"

def chat_with_gpt(prompt, history_list):
    
    response = openai.ChatCompletion.create(
        model = "gpt-4.1-mini",
        messages = [{"role":"user", "content": f"bu bizim mesajımız: {prompt}. Konuşma geçmişi: {history_list}"}]
        )
    
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    
    history_list = []
    
    while True:
        
        user_input = input("Kullanıcı tarafından girilen mesaj:")
        
        if user_input.lower() in ["exit","q"]:
            print("konusma tamamlandi")
            break
        history_list.append(user_input)
        response = chat_with_gpt(user_input, history_list)
        print(f"Chatboıt: {response}")