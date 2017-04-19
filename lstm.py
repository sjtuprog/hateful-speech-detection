
from __future__ import print_function
from __future__ import division
import json
import rnn
import datetime
import numpy as np
np.random.seed(1008)
import tensorflow as tf
tf.set_random_seed(1234)
import data_helpers 
from sklearn.model_selection import KFold 
import sys 
import h5py




embedding_model = data_helpers.loadGoogleModel('/Users/leigao/Documents/NLP/pre-trained-wordvec/GoogleNews-vectors-negative300.bin')

embedding_dim = len(embedding_model['dog'])
saveJ = './models/word_model.json'
saveW = './models/word_model_weights.h5' 



print('Loading data...')

data_set = data_helpers.loadJson('../corpus/comment/full-comments-u.json')


A = []
P = []
R = []
F = []
U = []

dense_outputs = 100 
batches = 128
sequence_length = 150

for fold in range(10):

	(train, test) = data_helpers.split_file(data_set, fold)

	(x_train, y_train) = data_helpers.encode_all(train,sequence_length,embedding_model)

	(x_test, y_test) = data_helpers.encode_all(test,sequence_length,embedding_model)

	model = rnn.attention_lstm_with_context(dense_outputs, sequence_length, embedding_dim)
	#model.summary()  
	model.fit(x_train,y_train,class_weight=None,batch_size=batches,validation_data=(x_test,y_test),epochs=30)
	y_predict = model.predict(x_test,batch_size=batches)
	
	pred = open('predictions/lstm-%d.txt' % fold , 'w')
    for i in range(len(y_predict)):
          pred.writelines(str(y_predict[i][0])+'\n')


	a,p,r,f,u = data_helpers.metrics(y_test,y_predict)
	print('Accuracy:\tPrecision:\tRecall:\tF-score:\tAUC:') 
	print('%f\t%f\t%f\t%f\t%f'%(a, p, r, f, u))	
	rnn.save_model(model,saveJ,saveW)
	A += [a]
    P += [p]
    R += [r]
    F += [f]
    U += [u]
print('Overall')
print('%f\t%f\t%f\t%f\t%f' %(sum(A)/len(A),sum(P)/len(P),sum(R)/len(R),sum(F)/len(F),sum(U)/len(U)))

 