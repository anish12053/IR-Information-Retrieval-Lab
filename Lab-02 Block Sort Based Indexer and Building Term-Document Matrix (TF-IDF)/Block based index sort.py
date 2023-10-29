import os
from math import *
from colorama import Fore
import time
import random
from wordcloud import STOPWORDS
from nltk.stem import PorterStemmer
import re
import os
import string

stops = set(STOPWORDS)

toStem = PorterStemmer()
set_of_unique_words = set()

print("Enter the path in which documents are present: ")
path = input()


def Remove(word):
    for i in word:
        if i in ".,!';:|-_()=/":
            word = word.replace(i,"")
    return word.lower()
list_of_created = []
list_of_available = []
for fileIs in os.listdir(path):
    fileIs = path + "/" + fileIs
    list_of_available.append(fileIs)
    with open(fileIs,'r',encoding='utf-8') as file:
        for line in file:
            for word in line.split():
                if word not in stops:
                    stemmed = toStem.stem(Remove(word))
                    if len(stemmed) > 2:
                        set_of_unique_words.add(toStem.stem(Remove(word)))
set_of_unique_words = sorted(set_of_unique_words)

block_Size = 100
current_file = 1
block = 0
dict_freq = {}
for every in set_of_unique_words:
    temp = 1
    for fileIs in os.listdir(path):
        fileIs = path + "/" + fileIs
        with open(fileIs,'r',encoding='utf-8') as file:
            if every in file.read():
                if every not in dict_freq:
                    dict_freq[every] = [temp]
                    block += 1
                else:
                    dict_freq[every].append(temp)
        temp += 1
    if block >= block_Size:
        with open(f"doc{current_file}.txt", "w") as f:
            list_of_created.append(f"doc{current_file}.txt")
            for value in dict_freq:
                f.write(value)
                f.write(" ")
                f.write(" ".join(str(e) for e in dict_freq[value]))
                f.write("\n")
        current_file += 1
        block = 1
        dict_freq.clear()

with open(f"doc{current_file}.txt","w") as f:
    list_of_created.append(f"doc{current_file}.txt")
    for value in dict_freq:
        f.write(value)
        f.write(" ")
        f.write(" ".join(str(e) for e in dict_freq[value]))
        f.write("\n")

block = 0
dict_freq.clear()

with open("merged.txt", "w") as new_created_file:
    for name in list_of_created:
        with open(name,"r") as file:
            for line in file:
                new_created_file.write(line)

for name in list_of_created:
    os.remove(name)
