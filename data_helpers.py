import numpy as np
np.random.seed(1008)
import random
random.seed(1280)
import tensorflow as tf
tf.set_random_seed(1234)
from gensim import models
from sklearn.utils import shuffle
from sklearn import metrics as SKM
import sys  
import random 
import re
import time 
import json  
from nltk import tokenize 

from sklearn.model_selection import KFold 

OOV = np.random.uniform(low=-0.25, high=0.25, size=(300,)) 

user_char_table = {}

 
def split_file(f_train,fold): 

    x_train = [] 
    y_train = []

    x_test = []
    y_test = []

    CV = 10

    kf = KFold(n_splits= CV , shuffle=True, random_state=123)
    T = []
    for train_index, test_index in kf.split(f_train):
        T += [(train_index, test_index)]

    train_index = T[fold][0]
    test_index  = T[fold][1]

    train_ = []
    test_ = []

    for i in train_index:
        train_ += [f_train[i]]


    for i in test_index:
        test_ += [f_train[i]]
 


     
    
    return train_, test_
 

 
 

def encode_text(x,sequence_length,embedding_model): # i = which dimension to encode
    global OOV
    embedding_dim = len(embedding_model['dog']) 
    x_train = np.zeros((len(x), sequence_length, embedding_dim))
    y_train = np.zeros(len(x))
    for dix, sent in enumerate(x): 
        counter = 0
        sent_array = np.zeros((sequence_length, embedding_dim))
        tokens = cleanStr(sent['text']).split()[:sequence_length] 
        for i in range(0,len(tokens)):
            t = tokens[i] 
            if(t in embedding_model):
                token_array = embedding_model[t]
            else:
                token_array = OOV
                
            sent_array[i, :] = token_array 
        x_train[dix, :, :] = sent_array
        y_train[dix] = int(sent['label'])

    return x_train,y_train

def encode_all(x,sequence_length,embedding_model):
    global OOV
    embedding_dim = len(embedding_model['dog'])
    x_train = np.zeros((len(x), sequence_length, embedding_dim))
    x_title = np.zeros((len(x), 15, embedding_dim))
    x_user = np.zeros((len(x),45,66))
    y_train = np.zeros(len(x))
    for dix, sent in enumerate(x):
        sent_array = np.zeros((sequence_length, embedding_dim))
        tokens = cleanStr(sent['text']).split()
        tokens = tokens[:sequence_length] 
        x_user[dix] = user_to_np(sent['user'])
        for i in range(0,len(tokens)):
            t = tokens[i]
            if(t in embedding_model):
                token_array = embedding_model[t]
            else:
                token_array = OOV

            sent_array[i, :] = token_array


        x_train[dix, :, :] = sent_array

        sent_title = np.zeros((15,embedding_dim))
        t_tokens = cleanStr(sent['title']).split()

        for i in range(0,len(t_tokens)):
            t = t_tokens[i]
            if(t in embedding_model):
                token_array = embedding_model[t]
            else:
                token_array = OOV

            sent_title[i, :] = token_array

        x_title[dix, :, :] = sent_title
        y_train[dix] = int(sent['label'])
    return [x_train,x_title,x_user], y_train

def user_to_np(s):
    global user_char_table
    a = np.zeros((45,66))
    for i in range(len(s)):
        v = np.zeros(66)
        c = s[i]
        if(c in user_char_table):
            v[ user_char_table[c] ] = 1
        else:
            v [ len(user_char_table)+1] = 1
            user_char_table[c] = len(user_char_table)+1
        a[i] = v
    return a

  


def loadJson(file):
    data = []
    
    with open(file,'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    return data

def cleanStr(string):  
    string = re.sub(r"['\"]+","",string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"ain\'t", " are not ", string)
    string = re.sub(r"\'m", " am ", string)
    string = re.sub(r"\'s", " is ", string) 
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"``",' \' ',string)
    string = re.sub(r"`",' \' ',string)
    string = re.sub(r"\s{2,}", " ", string) 
    return string.strip().lower()
 
 

def loadGloveModel(gloveFile):
    print "Loading Glove Model"
    start_time = time.clock()
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.asarray([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print "Done.",len(model)," words loaded!"

    end_time = time.clock()
    print 'Time in read %f s' %(end_time-start_time)
    return model

def loadGoogleModel(googleFile):
    print "Loading Google word2vec Model"
    start_time = time.clock()
    model = models.KeyedVectors.load_word2vec_format(googleFile, binary=True)

    end_time = time.clock()
    print 'Time in read %f s' %(end_time-start_time)
    return model  




def metrics(y_test,y_predict): 
    if(len(y_test)!=len(y_predict)):
        return None
    y_predict_b = [ ]
    y_predict_r = [ ]
    for i in range(len(y_test)):
        y_predict_b += [1 if(y_predict[i][0]>=0.5) else 0]
        y_predict_r += [y_predict[i][0]]

    acc = SKM.accuracy_score(y_test,y_predict_b)
    pre = SKM.precision_score(y_test,y_predict_b,pos_label=1)
    recall = SKM.recall_score(y_test,y_predict_b,pos_label=1)
    f = SKM.f1_score(y_test,y_predict_b,pos_label=1)
    auc = SKM.roc_auc_score(y_test,y_predict_r)
    return acc,pre,recall,f,auc



'''
def encode_multi_char(x,sequence_length,vocab, reverse_vocab, vocab_size, check, dim = 4):  
    embedding_dim = vocab_size
    input_data = np.zeros((len(x), dim, sequence_length, embedding_dim)) 
    for dix, sent in enumerate(x):  
        for i in range(0,dim):
            sent_array = np.zeros((sequence_length, embedding_dim))
            counter = 0 
            chars = list(sent[i])
            for c in chars:
                if counter >= sequence_length:
                    pass
                else:
                    char_array = np.zeros(vocab_size, dtype=np.int)
                    if c in check:
                        ix = vocab[c]
                        char_array[ix] = 1
                    sent_array[counter, :] = char_array
                    counter += 1 
            input_data[dix, i, :, :] = sent_array

    return input_data






def hybrid_generator_char(X_train, y,sequence_length,vocab, reverse_vocab, vocab_size, check,batch_size=32):
    order = range(len(X_train))
    while True:
        if not y is None:
            X_train, y = shuffle(X_train, y,random_state=130) 
            y = np.asarray(y)
        
        for i in xrange(np.ceil(1.0*len(X_train)/batch_size).astype(int)):
            
            #training set
            if not y is None: 
                x_train_batch = encode_multi_char(X_train[i*batch_size:(i+1)*batch_size],sequence_length,
                    vocab, reverse_vocab, vocab_size, check,4)    
                y_train_batch = y[i*batch_size:(i+1)*batch_size] 
                #print x_train_batch.shape
                yield x_train_batch, y_train_batch
                
            #test set
            else:
                x_train_batch = encode_multi_char(X_train[i*batch_size:(i+1)*batch_size],sequence_length,
                    vocab, reverse_vocab, vocab_size, check,4) 
                yield x_train_batch
def encode_onehot(x,sequence_length,vocab,dim=4):
    vocab_size = len(vocab)
    input_data = np.zeros((len(x), dim, sequence_length)) 
    for dix, sent in enumerate(x):  
        for i in range(0,dim):
            sent_array = np.zeros((sequence_length))
            counter = 0
            tokens = sent[i].split()
            for t in tokens:
                if counter >= sequence_length:
                    pass
                else: 
                    if(t in vocab):
                        token_array = vocab[t]
                    else:
                        token_array = -1
                    
                    sent_array[counter] = token_array
                    counter += 1
            input_data[dix, i, :] = sent_array

    return input_data

def random_generator(X_train, y,sequence_length,embedding_model,batch_size=32):
    order = range(len(X_train))
    while True:
        if not y is None:
            X_train, y = shuffle(X_train, y,random_state=130) 
            y = np.asarray(y)
        
        for i in xrange(np.ceil(1.0*len(X_train)/batch_size).astype(int)):
            if not y is None: 
                if(random.random()>0.95):
                    x_train_batch = encode_random(X_train[i*batch_size:(i+1)*batch_size], sequence_length,embedding_model)    
                    y_train_batch = np.array([0 for k in y[i*batch_size:(i+1)*batch_size]])
                else:
                    x_train_batch = encode_word(X_train[i*batch_size:(i+1)*batch_size], sequence_length,embedding_model)    
                    y_train_batch = y[i*batch_size:(i+1)*batch_size]
                yield x_train_batch, y_train_batch

            else:
                x_train_batch = encode_random(X_train[i*batch_size:(i+1)*batch_size], sequence_length,embedding_model)   
         
                yield x_train_batch
'''

'''



def embed_generator(X_train, y,sequence_length,embedding_model,batch_size=32): 
    while True:
        if not y is None:
            X_train, y = shuffle(X_train, y,random_state=130) 
            y = np.asarray(y)
        
        for i in xrange(np.ceil(1.0*len(X_train)/batch_size).astype(int)):
            
            #training set
            if not y is None:  
                x_train_batch = encode_word(X_train[i*batch_size:(i+1)*batch_size], sequence_length, embedding_model,0)
                y_train_batch = y[i*batch_size:(i+1)*batch_size] 
                yield x_train_batch, y_train_batch
                
            #test set
            else:
                x_train_batch = encode_word(X_train[i*batch_size:(i+1)*batch_size], sequence_length, embedding_model,0)
                yield x_train_batch
 





def hybrid_generator(X_train, y,sequence_length,embedding_model,batch_size=32):
    order = range(len(X_train))
    while True:
        if not y is None:
            X_train, y = shuffle(X_train, y,random_state=130) 
            y = np.asarray(y)
        
        for i in xrange(np.ceil(1.0*len(X_train)/batch_size).astype(int)):
            
            #training set
            if not y is None:  
                x_train_text = encode_word(X_train[i*batch_size:(i+1)*batch_size], 100, embedding_model,0) 
                x_train_prev = encode_word(X_train[i*batch_size:(i+1)*batch_size], 100, embedding_model,1) 
                x_train_succ = encode_word(X_train[i*batch_size:(i+1)*batch_size], 100, embedding_model,2)  
                L = [x_train_text,x_train_prev,x_train_succ]   
                y_train_batch = y[i*batch_size:(i+1)*batch_size]  
                yield L, y_train_batch
                
            #test set
            else:
                x_train_text = encode_word(X_train[i*batch_size:(i+1)*batch_size], 100, embedding_model,0) 
                x_train_prev = encode_word(X_train[i*batch_size:(i+1)*batch_size], 100, embedding_model,1) 
                x_train_succ = encode_word(X_train[i*batch_size:(i+1)*batch_size], 100, embedding_model,2)  
                L = [x_train_text,x_train_prev,x_train_succ]  
                yield L


def emotion_generator(X_train, y,sequence_length,embedding_model,batch_size=32):
    order = range(len(X_train))
    sid = SentimentIntensityAnalyzer()
    while True:
        if not y is None:
            X_train, y = shuffle(X_train, y,random_state=130) 
            y = np.asarray(y)
        
        for i in xrange(np.ceil(1.0*len(X_train)/batch_size).astype(int)):
            
            #training set
            if not y is None: 
                x_train_batch = encode_emotion(X_train[i*batch_size:(i+1)*batch_size], sequence_length, sid ,0)    
                y_train_batch = y[i*batch_size:(i+1)*batch_size]  
                yield x_train_batch, y_train_batch
                
            #test set
            else:
                x_train_batch = encode_emotion(X_train[i*batch_size:(i+1)*batch_size], sequence_length,sid,0)    
                yield x_train_batch


def encode_multi(x,sequence_length,embedding_model,dim = 4): 
    global OOV 
    embedding_dim = len(embedding_model['dog']) 
    input_data = np.zeros((len(x), dim, sequence_length, embedding_dim))
    for dix, sent in enumerate(x):  
        for i in range(0,dim):
            sent_array = np.zeros((sequence_length, embedding_dim))
            counter = 0
            tokens = cleanStr(sent[i]).split()
            for t in tokens:
                if counter >= sequence_length:
                    pass
                else: 
                    if(t in embedding_model):
                        token_array = embedding_model[t]
                    else:
                        token_array = OOV
                    
                    sent_array[counter, :] = token_array
                    counter += 1
            input_data[dix, i, :, :] = sent_array

    return input_data
'''