# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 20:52:26 2018

@author: kaiwend2
"""

import pandas as pd
from bs4 import BeautifulSoup
import warnings
train = pd.read_csv("~\\Downloads\\nlp_model\\popcorn\\labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("~\\Downloads\\nlp_model\\popcorn\\testData.tsv", header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv( "~\\Downloads\\nlp_model\\popcorn\\unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )

print("Read %d labeled train reviews, %d labeled test reviews, " \
 "and %d unlabeled reviews\n" % (train["review"].size,  
 test["review"].size, unlabeled_train["review"].size ))

replace_num = "[-+]?[0-9]*\.?[0-9]+"
replace_url = "http(s?)://[^\s]+"
#replace_lastURL="http(s?)://[^\s]+$"

# problem_sentences = []

def review_to_wordlist(raw_review):
    # remove HTML
#    with warnings.catch_warnings(record=True) as w:
#        warnings.simplefilter("always")    
    review_text = BeautifulSoup(raw_review).get_text()
        # assert problem_sentences.append(raw_review)
    # Replace http site to url
    url = re.sub(replace_url,"URL",raw_review)
    #url = re.sub(replace_lastURL,"URL",url)
    
    # Replace float to "NUM"
    num = re.sub(replace_num,"NUM",url)
    
    # remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", num)
    
    # Convert to lower case and split
    words = letters_only.lower().split()
    
    # Stopwords. Download them first!
    # from nltk.corpus import stopwords
    # stops = set(stopwords.words("english"))
    
    # Remove stop words
    # meaningful_words = [w for w in words if not w in stops]
    
    # Join the words back to string
    return(words)
    
# Download the punkt tokenizer for sentence splitting
import nltk
# nltk.download()

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def review_to_sentences(review):
    raw_sentences = tokenizer.tokenize(review.strip())
    
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence ))
    return sentences
#example = review_to_sentences(train["review"][243])
sentences = []
#i=0
for review in train["review"]:
    sentences += review_to_sentences(review)
    #print(i)
    #i+=1
print("Train data load complete")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review)
    
# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

