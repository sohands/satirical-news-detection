import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Merge, Embedding, Reshape, LSTM, GRU
from keras.optimizers import SGD, Adadelta
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling1D
from keras.layers.merge import average
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.callbacks import ModelCheckpoint
import cPickle
from auxillary import *
from attention_bias import *
import os
from random import shuffle

window_size=3
max_sent_length=100
word_embedding_size=300
required_sent_embedding_size=300
max_news_length=100
input_shape=(max_news_length,300,max_sent_length,1)
learning_rate=0.005
length_embedding_words=299527
lstm_num_hidden=150
# DATA_FOLDER='/media/sohan/Windows/Users/Sohan/Downloads/'
DATA_FOLDER='../'
trainX,trainY=loadTrainData(DATA_FOLDER+'train_data_X.npz',DATA_FOLDER+'train_data_Y.npz')
valX,valY=loadTrainData(DATA_FOLDER+'train_data_dev_X.npz',DATA_FOLDER+'train_data_dev_Y.npz')
#valdata=(valX,valY)
embedding_matrix=readObject('../embedding_matrix.pickle')
# embedding_matrix_fastext=readObject('../embedding_matrix_fastext.pickle')
# embedding_matrix=np.concatenate([embedding_matrix,embedding_matrix_fastext],axis=-1)
# syntax_X=np.load('syntax_X.npz')['arr_0']
syntax_X_val=np.load('syntax_X_dev.npz')['arr_0']
syntax_info_length=105 	
doc_syntax_dev=np.load('doc_syntax_dev.npz')['arr_0']
valX=[valX,syntax_X_val,doc_syntax_dev]
ordering=readObject('ordering.pickle')
# shuffle(ordering)
trainX=trainX[ordering]
trainY=trainY[ordering]
save_filename='models/cnn_dac_dbgru_da_te'
print 'Loaded'
# model=Sequential()
# conv_layer=Conv2D(filters=required_sent_embedding_size,kernel_size=(300,3),strides=(1,window_size))
# model.add(TimeDistributed(conv_layer,input_shape=input_shape))
# model.add(TimeDistributed(MaxPooling2D(pool_size=(1,13))))
# model.add(TimeDistributed(Flatten()))
# model.add(GlobalAveragePooling1D())
# model.add(Dropout(0.3))
# model.add(Dense(100,activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(2,activation='softmax'))

embedding_model=Sequential()
embedding_model.add(Reshape((max_news_length*max_sent_length,),input_shape=(max_news_length,max_sent_length,1)))
embedding_model.add(Embedding(length_embedding_words,word_embedding_size,weights=[embedding_matrix],
	input_length=max_news_length*max_sent_length,trainable=True))
syntax_model=Sequential()
syntax_model.add(Reshape((max_news_length*max_sent_length,11),input_shape=(max_news_length,max_sent_length,11)))
syntax_model.add(OneHotLayer(lengths=(44,16,16,9,3,7,2,2,2,2,2),required_output_shape=(max_news_length*max_sent_length,syntax_info_length)))
# syntax_model.summary()
# embedding_model.summary()
emb_model=Sequential()
emb_model.add(Merge([embedding_model,syntax_model],mode='concat'))
emb_model.add(Dense(150,activation='relu'))
emb_model.add(Dropout(0.1))
emb_model.add(Reshape((max_news_length,max_sent_length,word_embedding_size/2)))
emb_model.add(TimeDistributed(Conv1D(filters=required_sent_embedding_size,kernel_size=3)))
emb_model.add(TimeDistributed(MaxPooling1D(98)))
emb_model.add(TimeDistributed(Flatten()))
#model.add(Dropout(0.3))
emb_model.add(AttentiveConcatenateLayer(length=max_news_length,hidden_layers=(75,)))
emb_model.add(Bidirectional(GRU(lstm_num_hidden,return_sequences=True)))
emb_model.add(Bidirectional(GRU(lstm_num_hidden,return_sequences=True)))
emb_model.add(Bidirectional(GRU(lstm_num_hidden,return_sequences=True)))
emb_model.add(AttentionLayer(hidden_layers=(75,)))
emb_model.add(Dropout(0.3))
doc_syntax_model=Sequential()
doc_syntax_model.add(Reshape((105,),input_shape=(105,)))
model=Sequential()
model.add(Merge([emb_model,doc_syntax_model],mode='concat'))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2,activation='softmax'))
# model.summary()

optimizer=SGD(learning_rate,momentum=0.9,nesterov=True,clipnorm=1.)
metrics=['accuracy','categorical_crossentropy']
loss='categorical_crossentropy'
# checkpoint=ModelCheckpoint('bestmodels/cnn_dac_dbgru_da_te',monitor='val_acc',save_best_only=True,mode='max',verbose=1)

model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
print 'Compiled'
max_f1=0

for j in range(20):
	for i in range(0,len(trainX),10000):
		#print 'Training parts',i,'-',min(len(trainX),i+10000)
		syntax_X=np.load('../padded_syntax/syntax_X_'+str(min(len(trainX),i+10000))+'.npz')['arr_0']
		doc_syntax_X=np.load('./doc_syntax/syntax_X_'+str(min(len(trainX),i+10000))+'.npz')['arr_0']
		model.fit([trainX[i:min(len(trainX),i+10000)],syntax_X,doc_syntax_X],trainY[i:min(len(trainX),i+10000)],epochs=1,batch_size=20)
	# model.save('epochmodels/cnn_dac_dbgru_da_te_'+str(i))
	val_f1=F1(valY,model.predict(valX,batch_size=20))
	print 'Epoch',j,': Validation F1 score:',val_f1,
	if val_f1>max_f1:
		max_f1=val_f1
		model.save('bestmodels/cnn_dac_dbgru_dab_te_syntax_oh_trem_ds_f1')
		print 'Saved model'
	else:
		print ''
# model.fit(trainX,trainY,epochs=20,batch_size=32,validation_data=valdata,callbacks=[checkpoint])

# model.save(save_filename)
