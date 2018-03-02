import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora,models,similarities
import json
import random
import re
path="./dataset/review.json"
split_ratio = 0.8
subset_num = 100000

data = []
with open(path,'r',encoding='utf-8') as f:
    length_data = 0
    for line in f:
        length_data += 1

ids = list(range(length_data))
random.shuffle(ids)


train_id = ids[:int(0.8*length_data)]
test_id = ids[int(0.8*length_data):]

train_data = []
test_data = []
i=0
with open(path,'r',encoding='utf-8') as f:
    for line in f:
        content = json.loads(line)
        loading = dict()
        loading["text"] = content["text"]
        loading["stars"] = content["stars"]
        loading["id"] = i
        i += 1
        data.append(loading)


### Text Preprocessing
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
keep_stopwords = stopwords.words("english")[-36:]+stopwords.words("english")[131:133] + stopwords.words("english")[116:120]
new_stopwords = set(stopwords.words("english")).difference(set(keep_stopwords))
replace_num = "[-+]?[0-9]*\.?[0-9]+"
replace_url = "http(s?)://[^\s]+"
def review_to_words(raw_review):
    
    review_text = BeautifulSoup(raw_review).get_text()
    review_text = re.sub(r'[\t\r\n]'," ",review_text)
    url = re.sub(replace_url,"URL",raw_review)
    num = re.sub(replace_num,"NUM",url)
    letters_only = re.sub("[^a-zA-Z']", " ", num) #Also keep single quote
    
    words = letters_only.lower().split()
    
    meaningful_words = [w for w in words if not w in new_stopwords]
    
    return(" ".join(meaningful_words))

data = data[:subset_num]
clean_data = dict()    
clean_data["x"] = []
clean_data["y"] = []

import bar
step=0
total = len(data)
for each in data:
    each["text"] = review_to_words(each["text"])
    clean_data["x"].append(each["text"])
    clean_data["y"].append(each["stars"])
    bar.drawProgressBar(step/total)
    step +=1


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(clean_data["x"], clean_data["y"], test_size=0.33, random_state=42)
from gensim import corpora, models, similarities

# Make dictionary here. X_train = [["A","B","C"],["D","E"]]
dictionary = corpora.Dictionary(X_train)
dictionary.save('./tmp/X_train.dict')

corpus = [dictionary.doc2bow(text) for text in X_train]
corpora.MmCorpus.serialize('/tmp/X_train.mm', corpus)

tfidf = models.TfidfModel(corpus)

model = models.LdaModel(corpus, id2word=dictionary, num_topics=100)

import gensim