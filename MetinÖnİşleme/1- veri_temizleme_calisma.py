message= "Senniüüüi    Çoik ,  SeveriM !"

# %% Boşlukları kaldır.
message= "Senniüüüi   Çoik ,  SeveriM !"

message = " ".join(message.split())
print(f" Yeni mesajım : {message}")

# %% Noktalama işaretleri kaldıralım. 
import string
message= "Senniüüüi   Çoik ,  SeveriM !"

message = message.translate(str.maketrans("","",string.punctuation))
print(f" Yeni mesajım : {message}")


# %% Küçük harflere çevir.

message= "Senniüüüi   Çoik ,  SeveriM !"
message= message.lower()
print(f" Yeni mesajım : {message}")


# %% Özel işaretleri kaldırma
import re
message= "Senniüüüi $  Çoik ,  £ $ SeveriM !"

message = re.sub(r"[^A-Za-z0-9\s]", "", message) 
print(f" Yeni mesajım : {message}")

# %% Yazım hatalarını düzelt
from textblob import TextBlob
message= "Sennii   coik ,  SeveriM !"

message = TextBlob(message).correct()
print(f" Yeni mesajım : {message}")

# %% HTML Düzelt
from bs4 import BeautifulSoup
message= " <div>Senniüüüi   Çoik ,  SeveriM ! </div>"

message = BeautifulSoup(message,"html.parser").get_text()
print(f" Yeni mesajım : {message}")

