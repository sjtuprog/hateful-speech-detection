from __future__ import division 
import numpy as np
np.random.seed(1008) 
from sklearn.feature_extraction import FeatureHasher  
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold 
from sklearn import metrics as SKM
import pickle
import re
import nltk 
import time 
import json
from tqdm import tqdm 
from log_helper import *




D = read_LIWC('../corpus/liwc.dic')

L = loadEmotionLexicon('../corpus/NRC-emotion-lexicon.txt') 

def get_LIWC(t,label):
    global D
    C = {}
    z = np.zeros(125)
    words = cleanStr(t[label]).split()
    for w in words:
        if(w not in D): continue
        z = z + D[w]
            
    for i in range(0,125):
        c = label+' LIWC dim'+str(i)+'='+str(z[i])
        addDic(c,C)
    return C

def get_lexicon_emotion_feature(t,label):
    global L
    lexicon = L
    words = cleanStr(t[label]).split()
    s = np.asarray([0.0]*10)
    #count = 1.0
    for word in words:
        if(word.lower() in lexicon):
            s = s + lexicon[word.lower()]
      #count += 1.0
    #s = s/count

    d = {}
    d[label+'lexicon-emotion=anger']        = s[0]
    d[label+'lexicon-emotion=anticipation'] = s[1]
    d[label+'lexicon-emotion=disgust']      = s[2]
    d[label+'lexicon-emotion=fear']         = s[3]
    d[label+'lexicon-emotion=joy']          = s[4]
    d[label+'lexicon-emotion=negative']     = s[5]
    d[label+'lexicon-emotion=positive']     = s[6]
    d[label+'lexicon-emotion=sadness']      = s[7]
    d[label+'lexicon-emotion=surprise']     = s[8]
    d[label+'lexicon-emotion=trust']        = s[9]
    return d


def getCharNgram(text,label,n):
    C = {}
    for i in range(0,len(text)-n+1):
        c = 'char'+str(n)+'gram_of_'+label+'=' + text[i:i+n] 
        addDic(c,C)
    return C

def getWordNgram(text,label,n):
    C = {}
    words = cleanStr(text).split()
    for i in range(0,len(words)-n+1):
        c = 'word'+str(n)+'gram_of_'+label+'=' +  ' '.join(words[i:i+n])
        addDic(c,C)
    return C
 

def getChar(t,label):
    bi = getCharNgram(t[label],label,2)
    tri = getCharNgram(t[label],label,3)
    four = getCharNgram(t[label],label,4) 
    return mergeFeature(bi,tri,four)

def getWord(t,label):
    uni = getWordNgram(t[label],label,1)
    big = getWordNgram(t[label],label,2)
    return mergeFeature(uni,big)

def getNF(t,label): 
    c = getChar(t,label)
    w = getWord(t,label)

    l = get_LIWC(t,label)
    e = get_lexicon_emotion_feature(t,label) 
     

    d = mergeFeature(c,w,l,e) 


    return d



def textToDic(t):  
    text = getNF(t,'text')

    title = getNF(t,'title')   

    user = getNF(t,'user')
 
     
    d = mergeFeature(user) 
 
    return d






def cross_validation(): 
    tweets = loadJson('../corpus/comment/full-comments-u.json')
     

    x_train = []
    y_train = []
    
    for i in tqdm(range(0,len(tweets))):
        t = tweets[i]
        x_train += [ textToDic(t)   ] 
        y_train += [ int(t['label']>0)   ]   

    features = getFeatureSize(x_train)

    hasher = FeatureHasher(input_type='dict',n_features = features)
    
    print 'we have %d features' % features



    X = hasher.transform(x_train)
    y = np.asarray(y_train)
    
    clf = LogisticRegression(C=1.0, dual=False, fit_intercept=True, intercept_scaling=1, class_weight='balanced', penalty='l2',n_jobs=4)
 
    splits = 10

    kf = KFold(n_splits= splits,shuffle=True, random_state=123)
    split = 0

    A = []
    P = []
    R = []
    F = []
    T = []
    U = []
    for train_index, test_index in kf.split(X):
        T += [(train_index, test_index)]
 

    print 'Fold:    Acc:    Prec:   Reca:   F-1:    '
    for i in range(0,splits):
        train_index = T[i][0]
        test_index = T[i][1]
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train,y_train)
        y_predict = clf.predict_proba(X_test)
        a,p,r,f,u = metrics(y_test,y_predict) 
        A.append(a)
        P.append(p)
        R.append(r)
        F.append(f)
        U.append(u)
        
        pred = open('predictions/log-%d.txt' % split ,'w')
        for i in range(len(y_predict)):
            pred.writelines(str(y_predict[i][1])+'\n')

        print '%d\t%f\t%f\t%f\t%f\t%f' % (split,a,p,r,f,u) 
        split+=1
 
    print 'Overall\t%f\t%f\t%f\t%f\t%f' %(sum(A)/len(A),sum(P)/len(P),sum(R)/len(R),sum(F)/len(F),sum(U)/len(U))
 

cross_validation()