import nltk
nltk.download('punkt')
from nltk import word_tokenize

f = "Hi, This is Koushik"
tokens = word_tokenize(f)
print(tokens)