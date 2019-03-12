import nltk
from itertools import zip_longest
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from PyRouge.pyrouge import Rouge

r = Rouge()
list = []


def accuracy():
    with open("C:/Users/kiree/Desktop/UMKC/2nd Semester/Big Data Analytics and Applications/ICP/ICP6/Tutorial 6 Source Code/medium-show-and-tell-caption-generator-master/etc/predict.txt") as f2,  open("C:/Users/kiree/Desktop/UMKC/2nd Semester/Big Data Analytics and Applications/ICP/ICP6/Tutorial 6 Source Code/medium-show-and-tell-caption-generator-master/etc/true.txt") as f1:
        k = 0
        for line in f2:
            list.append(line)
        for line1 in f1:
            i = 0
            y_true = list[i]
            y_true_line = []
            i= i+1
            y_pred = (line1)
            BLEUscore = sentence_bleu(word_tokenize(y_true), word_tokenize(y_pred), weights=(1, 0, 0, 0))
            [precision, recall, f_score] = r.rouge_l([y_true], [y_pred])
            print("Precision is :" + str(precision) + "\nRecall is :" + str(recall) + "\nF Score is :" + str(f_score))
            print(BLEUscore)


if __name__ == "__main__":
    accuracy()