import nltk
from nltk import word_tokenize
import linecache
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
import matplotlib.pyplot as plot
plot.rcdefaults()
import numpy as np



linecounter = 0
Chicken_count = 0
Coke_count = 0
Vegetable_count = 0

f = open('Extracted_tokens.txt','a')
f1 = open('Extracted_urls.txt','a')
punctuations = "?:!.,;"
stop_words = set(stopwords.words('english'))
with open('SBU_captioned_photo_dataset_captions.txt','r') as file:
    for line in file:
        linecounter = linecounter + 1
        tokens = word_tokenize(line)
        for word in tokens:
            if word in punctuations:
                tokens.remove(word)
            if word in stop_words:
                tokens.remove(word)
            #new_tokens = set(tokens)
        for word in tokens:
            lemmit_word = wordnet_lemmatizer.lemmatize(word, pos="v")
            if 'Chicken' == lemmit_word:
                Chicken_count = Chicken_count + 1
                f.writelines(line)
                line1 = linecache.getline('SBU_captioned_photo_dataset_urls.txt', linecounter)
                f1.writelines(line1)
            if 'Coke' == lemmit_word:
                Coke_count = Coke_count + 1
                f.writelines(line)
                line1 = linecache.getline('SBU_captioned_photo_dataset_urls.txt', linecounter)
                f1.writelines(line1)
            if 'Vegetable' == lemmit_word:
                Vegetable_count = Vegetable_count + 1
                f.writelines(line)
                line1 = linecache.getline('SBU_captioned_photo_dataset_urls.txt', linecounter)
                f1.writelines(line1)
objects = ('Chicken', 'Coke', 'Vegetable')
y_pos = np.arange(len(objects))
performance = [Chicken_count,Coke_count,Vegetable_count]
plot.bar(y_pos, performance, align='center', alpha=0.5)
plot.xticks(y_pos, objects)
plot.show()
