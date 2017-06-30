import numpy as np
import random
np.random.seed(1008)
import tensorflow as tf
tf.set_random_seed(1234)
import json
import keras  
from keras.preprocessing import sequence
from keras.utils import np_utils 
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D, LSTM
from keras import metrics 
from keras import backend as K
K.set_learning_phase(1)
from keras.regularizers import l2
from keras.callbacks import *
# from visualizer import *
from keras.models import * 
from keras.layers.core import *
from keras.layers import Input, Embedding, LSTM, Dense, merge, TimeDistributed ,Bidirectional
#for attention ,see https://github.com/philipperemy/keras-attention-mechanism

def save_model(model,saveJ,saveW):
	json_string = model.to_json()
	with open(saveJ, 'w') as f:
	    json.dump(json_string, f)
	model.save_weights(saveW)  

def read_model(saveJ,saveW):
	json_file = open(saveJ, 'r')
	loaded_model_json = json.load(json_file)
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(saveW)
	print "Loaded model from disk"+ saveJ+' '+ saveW 
	loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return loaded_model  

def lstm(lstm_outputs,sequence_length,embedding_dim):
	print('Build model...')
	main_input = Input(shape=(sequence_length, embedding_dim)) 
	bilstm = LSTM(lstm_outputs,return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(main_input)  
	out = Dense(1, activation='sigmoid')(bilstm)
	model = Model(input=main_input, output= out) 
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
 

def blstm(lstm_outputs,sequence_length,embedding_dim):
	print('Build model...')
	main_input = Input(shape=(sequence_length, embedding_dim)) 
	bilstm = Bidirectional(LSTM(lstm_outputs,return_sequences=False, dropout=0.2, recurrent_dropout=0.2))(main_input)  
	out = Dense(1, activation='sigmoid')(bilstm)
	model = Model(input=main_input, output= out) 
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def attention_lstm(lstm_outputs,sequence_length, embedding_dim): 
 
	len_text = sequence_length 

	lstm_units = lstm_outputs

	in_text = Input(shape=(len_text, embedding_dim))  
	bilstm = Bidirectional(LSTM(lstm_units,return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(in_text)  
	attention = TimeDistributed(Dense(1,activation='tanh'))(bilstm)
	attention = Flatten()(attention)
	attention = Activation('softmax')(attention)
	attention = RepeatVector(2*lstm_units)(attention)
	attention = Permute([2, 1])(attention)
	sent_representation = merge([bilstm, attention], mode='mul')
	out_text = Lambda(lambda xin: K.sum(xin, axis= 1), output_shape=(2*lstm_units,), name='sent')(sent_representation) 
 
	output_ = Dense(1, activation='sigmoid')(out_text) 

	model = Model(input=in_text, output= output_) 
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model 
 

def attention_lstm_with_context(lstm_units,len_text, embedding_dim):


    in_text = Input(shape=(len_text, embedding_dim))
    bilstm = Bidirectional(LSTM(lstm_units,return_sequences=True, recurrent_dropout=0.2))(in_text)
    attention = TimeDistributed(Dense(1,activation='tanh'))(bilstm)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(2*lstm_units)(attention)
    attention = Permute([2, 1])(attention)
    sent_representation = merge([bilstm, attention], mode='mul')
    out_text = Lambda(lambda xin: K.sum(xin, axis= 1), output_shape=(2*lstm_units,), name='text')(sent_representation)

    in_title = Input(shape=(15, embedding_dim))
    out_title  = Bidirectional(LSTM(lstm_units,return_sequences= False, recurrent_dropout=0.2))(in_title)

    in_user = Input(shape=(45,66))
    out_user = Bidirectional(LSTM(lstm_units,return_sequences= False, recurrent_dropout=0.2))(in_user)

    outs = merge([out_text, out_title, out_user], mode='concat')

    output_ = Dense(1, activation='sigmoid')(outs)

    model = Model(input=[in_text,in_title,in_user], output= output_)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
