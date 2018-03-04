from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.sparse import csr_matrix,hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
import re
import string
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
import numpy as np
from collections import Counter
import time


class FeatureExtractor(TransformerMixin, BaseEstimator):

    def __init__(self, factor):
        self.factor = factor

    def transform(self, data):
        return data[self.factor].values

    def fit(self, *_):
        return self


class ImputeNA(TransformerMixin, BaseEstimator):
    def __init__(self, na_replacement=None):
        if na_replacement is not None:
            self.NA_replacement = na_replacement
        else:
            self.NA_replacement = 'missing'

    def transform(self, data):
        data = pd.DataFrame(data)
        data = data.fillna(self.NA_replacement).values
        return np.ascontiguousarray(data)

    def fit(self, *_):
        return self


class CategoricalEncoding(TransformerMixin, BaseEstimator):

    def __init__(self, method=None):
        if method is not None:
            self.encoding_method = method
        else:
            self.encoding_method = 'OneHot'

    def transform(self, data):
        le = LabelEncoder()
        enc = OneHotEncoder()
        b = csr_matrix

        if self.encoding_method == 'OneHot':
            for i in range(data.shape[1]):
                tmp = le.fit_transform(data[:,i]).reshape(-1, 1)
                if i==0:
                    b = hstack([enc.fit_transform(tmp)])
                else:
                    b = hstack([b,enc.fit_transform(tmp)])
        elif self.encoding_method == 'Numerical':
            for i in range(data.shape[1]):
                tmp = le.fit_transform(data[:,i]).reshape(-1, 1)
                tmp = csr_matrix(tmp)
                if i==0:
                    b = hstack([tmp])
                else:
                    b = hstack([b,tmp])
        return b

    def fit(self, *_):
        return self

class text(TransformerMixin, BaseEstimator):
    def __init__(self, method=None, ngram = None, max_f = None, binary = None, stopwords=None,tokenizer=None,analyzer =None):
        # method
        if method is not None:
            self.transformmethod = method
        else:
            self.transformmethod = 'cv'
        # ngram
        if ngram is not None:
            self.ngram = ngram
        else:
            self.ngram = 1
        # max_f
        if max_f is not None:
            self.maxf = max_f
        else:
            self.maxf = 5000
        # binary
        if binary is not None:
            self.bi = binary
        else:
            self.bi = False
        # stopwords
        if stopwords is not None:
            self.stopwords = stopwords
        else:
            self.stopwords = None
        # tokenizer
        if tokenizer is not None:
            self.token = tokenizer
        else:
            self.token = None
        
        # analyzer
        if analyzer is not None:
            self.analyzer = analyzer
        else:
            self.analyzer = 'word'
        
    def transform(self, data):
        if self.transformmethod == 'cv':
            cv = CountVectorizer(tokenizer=self.token,
                                 stop_words=self.stopwords,
                                 ngram_range=(1,self.ngram),
                                 max_features = self.maxf,
                                 binary = self.bi,
                                 analyzer = self.analyzer)
            return cv.fit_transform(data)
        elif self.transformmethod == 'tfidf':
            tfidf = TfidfVectorizer(tokenizer=self.token,
                                    stop_words=self.stopwords,
                                    ngram_range=(1,self.ngram),
                                    max_features = self.maxf,
                                    binary = self.bi,
                                    analyzer = self.analyzer)
            return tfidf.fit_transform(data)

    def fit(self, *_):
        return self
    
    

class text_to_seq(TransformerMixin, BaseEstimator):
    def __init__(self, name=None,total_text=None):
        self.name = name
        self.text = total_text

    def transform(self, data):
        data_seq = self.text.texts_to_sequences(data)
        if self.name=='name':
            for i in data_seq:
                while len(i) <12:
                    i.append(0)
            data_seq = np.array(data_seq)
        elif self.name == 'desc':
            for i in data_seq:
                while len(i)<120:
                    i.append(0)
                while len(i)>120:
                    del i[120:]
            data_seq = np.array(data_seq)

        return data_seq

    def fit(self, *_):
        return self

class optimized_categorical_numerical_encoding(TransformerMixin, BaseEstimator):
    def __init__(self, min_df):
        self.mindf = min_df

    def transform(self, data):
        a = pd.DataFrame(data.reshape(1,-1)[0],columns=['brand_name'])
        data_dic = Counter(a.values[a.values != 'missing'])
        data_list = sorted(b for (b, c) in data_dic.items() if c >= self.mindf)
        data_idx = {b: (i + 1) for (i, b) in enumerate(data_list)}
        x_data = a.brand_name.apply(lambda b: data_idx.get(b, 0))
        x_data = x_data.values.reshape(-1, 1)
        return x_data

    def fit(self, *_):
        return self

    

def transform_text_func(method=None, ngram = None, max_f = None, binary = None, stopwords=None,token=None,analyzer =None):
    if method == 'cv':
        return CountVectorizer(tokenizer=token,
                               stop_words= stopwords,
                               ngram_range=(1, ngram),
                               max_features = max_f,
                               binary = binary,
                               analyzer = analyzer)
    elif method == 'tfidf':
        return TfidfVectorizer(tokenizer=token,
                               stop_words= stopwords,
                               ngram_range=(1, ngram),
                               max_features = max_f,
                               binary = binary,
                               analyzer = analyzer)

def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
