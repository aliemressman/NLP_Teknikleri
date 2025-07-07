"""
spam veriseti -> spam ve ham -> binary classfication with Decision Tree
"""

# import libraries
import pandas as pd

# verisetini yukle
data = pd.read_csv("spam.csv", encoding= "latin-1")
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis =1, inplace =True)
data.columns = ["label","text"]

# EDA: Kesifsel Veri Analizi: missing value(kayip deger)
print(data.isna().sum())


# %% text cleaning and text preprocessing: ozel karakterler, lowercase, tokenization, stopwords, lemmazite
import nltk
nltk.download("stopwords") # cok kullanilan ve anlam tasimayan sozcukleri metin icerisinden cikartalim
nltk.download("wordnet") # Lemma bulmak icin gerekli veriseti
nltk.download("omw-1.4") # wordnete ait farkli dillerin kelime anlamlarini iceren veriseti

import re 
from nltk.corpus import stopwords # stopwords kurtulmka icin
from nltk.stem import  WordNetLemmatizer # lemmazitaion

text = list(data.text)
lemmaziter = WordNetLemmatizer()

corpus = []

for i in range(len(text)):
    
    r = re.sub("[^a-zA-Z]", " ", text[i]) # metin icerisinde harf olmayan tum karakterlerden kurtul
    
    r = r.lower() # kucuk harfe cevir
    
    r = r.split() # kelimeleri ayir
    
    r = [word for word in r if word not in stopwords.words("english")] # stopwordslerden kurtul
    
    r = [lemmaziter.lemmatize(word) for word in r]
    
    r = " ".join(r)
    
    corpus.append(r)

data["text2"] = corpus 


# %% model training and evaluation 

X = data["text2"]
y = data["label"] # target variable

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state=42)

# feature extraction : Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
deneme = X_train_cv.toarray()

# classifier training: model training and evaluation 
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train_cv, y_train) # egitim

X_test_cv = cv.transform(X_test)

# prediction
prediction = dt.predict(X_test_cv)

from sklearn.metrics import confusion_matrix
c_matrix = confusion_matrix(y_test, prediction)

accuracy = 100*(sum(sum(c_matrix)) - c_matrix[1,0] - c_matrix[0,1])/ sum(sum(c_matrix))

print(f"basari orani: {accuracy}")











