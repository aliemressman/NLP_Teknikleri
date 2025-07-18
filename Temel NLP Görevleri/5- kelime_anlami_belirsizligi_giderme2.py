from pywsd.lesk import simple_lesk, adapted_lesk, cosine_lesk

# ornek cumle
sentences = [
    "I go to the bank to deposit money",
    "The river bank was flooded after the heavy rain"]

word = "bank"

for s in sentences:
    print(f"sentence: {s}")
    
    sense_simple_lesk = simple_lesk(s, word)
    print(f"Sense simple: {sense_simple_lesk.definition()}")
    
    sense_adapted_lesk = adapted_lesk(s, word)
    print(f"Sense simple: {sense_adapted_lesk.definition()}")
    
    
    sense_cosine_lesk = cosine_lesk(s, word)
    print(f"Sense simple: {sense_cosine_lesk.definition()}")