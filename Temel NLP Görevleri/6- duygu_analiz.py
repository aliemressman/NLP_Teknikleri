"""
Problem tanimi ve veriseti: 

    amazon veri seti icerisinde bulunan yorumlarin positive mi negartive mi oldugunu siniflanidirmak 
    binary classification problemi
"""

# import libraries
import pandas as pd
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

# veri seti yukle
df = pd.read_csv("duygu_analizi_amazon_veri_seti.csv")

# text cleaning ve preprocessing
lemmatizer = WordNetLemmatizer()
def clean_preprocess_data(text):
    
    # tokenize
    tokens = word_tokenize(text.lower())
    
    # stopwords
    filtered_tokens = [token for token in tokens if token not in stopwords.words("english")]
    
    # lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    # join words
    processed_text = " ".join(lemmatized_tokens)
    
    return processed_text

df["reviewText2"] = df["reviewText"].apply(clean_preprocess_data)

# sentiment analysis (nltk)
analyzer = SentimentIntensityAnalyzer()

def get_sentiments(text):
    
    score = analyzer.polarity_scores(text)
    
    sentiment = 1 if score["pos"] > 0 else 0
    
    return sentiment

df["sentiment"] = df["reviewText2"].apply(get_sentiments)


# evaluation - test
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(df["Positive"], df["sentiment"])
print(f" Confusion matrix: {cm}")

cr = classification_report(df["Positive"], df["sentiment"])
print(f" classification_report : \n{cr}")










