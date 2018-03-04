
# coding: utf-8

# In[ ]:


import tensorflow as tf
import pandas as pd
import re
train = pd.read_csv("./training_data.csv", header=0, delimiter="\t|\n")
data = train["comment"]
### Text Preprocessing
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
    letters_only = re.sub("[^a-zA-Z#]", " ", num) #Also keep single quote
    
    words = letters_only.lower().split()
    
    meaningful_words = [w for w in words if not w in new_stopwords]
    
    return(" ".join(meaningful_words))

clean_data = dict()    
clean_data["x"] = []
clean_data["y"] = []
import bar
step=0
total = len(data)
for each in data:
    clean_data["x"].append(review_to_words(each))
    bar.drawProgressBar(step/total)
    step +=1


clean_data["y"] = train["score"]


# In[ ]:


import numpy as np
def int2vector(i):
    return np.array([1]*(i+1)+[0]*(10-i)).reshape(11,1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(clean_data["x"], clean_data["y"], test_size=0.33, random_state=42)

y_train = [int2vector(i) for i in y_train]


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer 
def to_predict(row):
        for i in range(11):
            if row[-1]>0.5:
                return 10
            elif row[i]<0.5:
                return i-1

logging = {"i":0,"epoch":0,"epochs":50}            
            
def generate_batch(features,labels,ids):
    global logging
    if logging["i"]==len(ids)-1:
        inner_i = logging["i"]
        logging["i"] = 0
        logging["epoch"] += 1
        return features[ids[inner_i]:],labels[ids[inner_i]:]
    else:
        inner_i = logging["i"]
        logging["i"] += 1
        return features[ids[inner_i]:ids[logging["i"]]],labels[ids[inner_i]:ids[logging["i"]]]



def sum_to_pred(row):
    r = np.rint(np.sum(row))
    if r>10:
        return 10
    elif r<0:
        return 0
    else:
        return r
        
def tuning(num_features,H,batch_size):   
    global logging
    logging["i"] = 0
    logging["epoch"] = 0
    ids = list(range(0,len(X_train),batch_size))
    print("feature:%i,H:%i,batch:%i"%(num_features,H,batch_size))
    #num_features = 5000
    vectorizer = CountVectorizer(analyzer = "word",                                    tokenizer = None,                                     preprocessor = None,                                  stop_words = None,                                    max_features = num_features) 
    train_data_features = vectorizer.fit_transform(X_train)
    train_data_features = train_data_features.toarray()
    train_data = [train_data_features[i,] for i in range(train_data_features.shape[0])]
    
    test_data_features = vectorizer.transform(X_test)
    test_data_features = test_data_features.toarray()
    test_data = [test_data_features[i,] for i in range(test_data_features.shape[0])]

    #batch_size = 10

    #H = 5
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32,shape=[None,num_features])
        y_ = tf.placeholder(tf.float32,shape=[None,11,1])
        new_y = tf.squeeze(y_)
        W_1 = tf.Variable(tf.random_uniform([num_features,H]))
        b_1 = tf.Variable(tf.zeros([H]))

        W_2 = tf.Variable(tf.random_uniform([H,11]))
        b_2 = tf.Variable(tf.zeros([11]))

        temp_1 = tf.nn.relu(tf.matmul(x,W_1) + b_1)
        temp_2 = tf.matmul(temp_1,W_2) + b_2
        output = tf.nn.sigmoid(temp_2)
        #loss = tf.reduce_mean(tf.square(tf.subtract(output,new_y)))
        loss = tf.losses.mean_squared_error(output,new_y)
        opt = tf.train.AdamOptimizer(learning_rate=0.01)
        opt_op = opt.minimize(loss)
        init = tf.global_variables_initializer()

    # with tf.Session(graph = graph) as session:

    sess = tf.InteractiveSession(graph=graph)
    sess.run(init)
    print("Initialized")
    step = 0 
    while logging["epoch"]<logging["epochs"]:
        step+=1
        batch_x,batch_y = generate_batch(train_data,y_train,ids)
        feed_dict={x:batch_x,y_:batch_y}
        _,loss_val = sess.run([opt_op,loss],feed_dict=feed_dict)
        if step % 2000 == 0:
            print("Step: %i, Epoch: %i, Loss:%f"%(step,logging["epoch"],loss_val))

    result = sess.run(output,feed_dict={x:test_data_features})
    result_label = [to_predict(each) for each in result]
    from sklearn.metrics import mean_squared_error
    sd = mean_squared_error(result_label, y_test)


    result = sess.run(output,feed_dict={x:test_data_features})
    result_label = [sum_to_pred(each) for each in result]
    from sklearn.metrics import mean_squared_error
    custom = mean_squared_error(result_label, y_test)
    sess.close()
    return sd,custom


# In[ ]:


tuning_feature = [2000,5000,8000]
tuning_H = [2,5,8,10]
batch_size = [300,100,50]

BIG = dict()
for a in tuning_feature:
    for b in tuning_H:
        for c in batch_size:
            first,second = tuning(a,b,c)
            BIG["feature:%i,H:%i,batch:%i"%(a,b,c)] = [first,second]
            print(BIG)
