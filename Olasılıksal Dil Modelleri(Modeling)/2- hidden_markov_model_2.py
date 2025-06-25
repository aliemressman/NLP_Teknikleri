# import libraries
import nltk
from nltk.tag import hmm
from nltk.corpus import conll2000


# gerekli verisetini iceri aktar
# nltk.download("conll2000") 1 kere indirmek yetiyor

train_data = conll2000.tagged_sents("train.txt")
test_data = conll2000.tagged_sents("test.txt")

print(f"train_data: {train_data[:1]}")

# train hmm
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)


# yeni cumle ve test
test_sentence = "I like going to school".split()
tags = hmm_tagger(test_sentence)