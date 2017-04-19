from __future__ import division 
import numpy as np
np.random.seed(1008) 
from sklearn.feature_extraction import FeatureHasher  
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold 
from sklearn import metrics as SKM 
import re
import nltk 
import time 
import json


def metrics(y_test,y_predict): 
    y_predict_b = [ ]
    y_predict_r = [ ]
    for i in range(len(y_test)):
        y_predict_b += [1 if(y_predict[i][1]>0.5) else 0]
        y_predict_r += [y_predict[i][1]]

    acc = SKM.accuracy_score(y_test,y_predict_b)
    pre = SKM.precision_score(y_test,y_predict_b,pos_label=1)
    recall = SKM.recall_score(y_test,y_predict_b,pos_label=1)
    f = SKM.f1_score(y_test,y_predict_b,pos_label=1)
    auc = SKM.roc_auc_score(y_test,y_predict_r)
    return acc,pre,recall,f,auc



def cleanStr(string):  
    string = re.sub(r"['\"]+","",string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
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
    string = re.sub(r"`",'',string)
    string = re.sub(r"\s{2,}", " ", string) 
    return string.strip().lower()


def addDic(e,d):
    if(e not in d):
        d[e]=1
    else:
        d[e]+=1

def loadJson(file):
    data = []
    with open(file,'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def getFeatureSize(X):
    d = {}
    i=0
    for x in X:
        for c in x:
            if(c not in d):
                d[c] = i
                i+=1
    return len(d)   

def read_LIWC(liwc):
    f = open(liwc,'r')
    d = {}
    for line in f:
        z = line.strip().split()
        q = np.zeros(125)
        word = z[0]
        for i in z:
            if(re.match('^[0-9]+$',i)):  
                q[int(i)-1] = 1
        d[word] = q
    return d

def loadEmotionLexicon(emotionFile):
  print 'Load emotion lexicon'
  f = open(emotionFile,'r')
  lexicon = {}

  for line in f:
    splitLine = line.split('\t')
    if( splitLine[0] not in lexicon):
      lexicon[splitLine[0]] = [int(splitLine[2])]
    else:
      lexicon[splitLine[0]] += [int(splitLine[2])]
  for word in lexicon:
    lexicon[word] = np.asarray(lexicon[word])
    
  return lexicon
 
def mergeFeature(*features):
    d = {}
    for f in features:
        for j in f:
            addDic(j,d)
    return d
    