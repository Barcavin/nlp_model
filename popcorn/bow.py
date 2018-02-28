import pandas as pd
train = pd.read_csv("~\\Downloads\\nlp_model\\popcorn\\labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

from bs4 import BeautifulSoup
# example1 = BeautifulSoup(train["review"][0])
import re
from nltk.corpus import stopwords

def review_to_words(raw_review):
    # remove HTML
    review_text = BeautifulSoup(raw_review).get_text()
    
    # remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    
    # Convert to lower case and split
    words = letters_only.lower().split()
    
    # Stopwords. Download them first!
    # from nltk.corpus import stopwords
    stops = set(stopwords.words("english"))
    
    # Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    
    # Join the words back to string
    return(" ".join(meaningful_words))
    

clean_train_review = []
for i in range(train["review"].size):
    clean_train_review.append(review_to_words(train["review"][i]))
    
from sklearn.feature_extraction.text import CountVectorizer  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

train_data_features = vectorizer.fit_transform(clean_train_review)
train_data_features = train_data_features.toarray()
vocab = vectorizer.get_feature_names()

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100) 
forest = forest.fit( train_data_features, train["sentiment"] )