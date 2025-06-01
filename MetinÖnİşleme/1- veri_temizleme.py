# Metinlerde bulunan fazla boşlukları kaldır
text = "Hello,       World!      2035"

"""
text.split()
Out[3]: ['Hello,', 'World!', '2035']
"""

cleaned_text1 = " ".join(text.split())
print(f"text: {text} \n cleaned_text1: {cleaned_text1}")

# %% Büyük -> Küçük harf çevirimi
text = "Hello, World 2035"
cleaned_text2 = text.lower() # Küçük harfe çevir.
print(f"text: {text} \n cleaned_text2: {cleaned_text2}")

# %% Noktalama işaretlerini kaldır.
import string

text = "Hello, World! 2035"
cleaned_text3 = text.translate(str.maketrans("", "",string.punctuation))
print(f"text:{text} \n cleaned_text3: {cleaned_text3}")

# %% Özel Karakterleri kaldır, %, @, /, +, ^
import re

text = "Hello, World! 2035%"
cleaned_text4 = re.sub(r"[^A-Za-z0-9\s]","",text)
print(f"text:{text} \n cleaned_text4: {cleaned_text4}")

# %% Yazım hatalarını duzelt
from textblob import TextBlob # Metin analizlerinde kullanılan bir kütüphane

text = "Hellıo, Wirld! 2035"
cleaned_text5 = TextBlob(text).correct() # correct: Yazim hatalarını düzeltir.
print(f"text:{text} \n cleaned_text5: {cleaned_text5}")

# %% HTML ya da URK etiketlerini kaldırma.
from bs4 import BeautifulSoup

html_text = "<div>Hello, World! 2035</div>" # html etiketi var
# Beautiful Soup ile html yapısını parse et, get_text ile text kismini çek.
cleaned_text6 = BeautifulSoup(html_text,"html.parser").get_text()
print(f"text:{html_text} \n cleaned_text6: {cleaned_text6}")
