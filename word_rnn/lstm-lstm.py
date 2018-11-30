import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Merge, Embedding, Reshape, GRU
from keras.optimizers import SGD, RMSprop
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling1D
from keras.layers.merge import average
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.callbacks import ModelCheckpoint
import cPickle
from auxillary import *

window_size=3
max_sent_length=100
word_embedding_size=300
required_sent_embedding_size=300
max_news_length=100
input_shape=(max_news_length,300,max_sent_length,1)
learning_rate=0.05
length_embedding_words=299527
sent_lstm_size=150
# DATA_FOLDER='/media/sohan/Windows/Users/Sohan/Downloads/'
DATA_FOLDER='../'
trainX,trainY=loadTrainData(DATA_FOLDER+'train_data_X.npz',DATA_FOLDER+'train_data_Y.npz')
valX,valY=loadTrainData(DATA_FOLDER+'train_data_dev_X.npz',DATA_FOLDER+'train_data_dev_Y.npz')
valdata=(valX,valY)
embedding_matrix=readObject('../embedding_matrix.pickle')
save_filename='models/dbgru_dbgru_te_1'

model=Sequential()
model.add(Reshape((max_news_length*max_sent_length,),input_shape=(max_news_length,max_sent_length,1)))
model.add(Embedding(length_embedding_words,word_embedding_size,weights=[embedding_matrix],
	input_length=max_news_length*max_sent_length,trainable=False))
model.add(TimeDistributed(Dense(300,activation='linear')))
model.add(Reshape((max_news_length,max_sent_length,word_embedding_size)))
model.add(TimeDistributed(Bidirectional(GRU(required_sent_embedding_size/2,return_sequences=True))))
model.add(TimeDistributed(Bidirectional(GRU(required_sent_embedding_size/2))))
model.add(Bidirectional(GRU(sent_lstm_size,return_sequences=True)))
model.add(Bidirectional(GRU(sent_lstm_size)))
model.add(Dropout(0.3))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2,activation='softmax'))
model.summary()

optimizer=SGD(learning_rate,decay=1e-6,momentum=0.9,nesterov=True,clipnorm=1.)
#optimizer=RMSprop(learning_rate)
metrics=['accuracy','categorical_crossentropy']
loss='categorical_crossentropy'
checkpoint=ModelCheckpoint('bestmodels/dbgru_dbgru_te_1',monitor='val_acc',save_best_only=True,mode='max',verbose=1)

model.compile(optimizer=optimizer,loss=loss,metrics=metrics)

model.fit(trainX,trainY,epochs=20,batch_size=20,validation_data=valdata,callbacks=[checkpoint])

model.save(save_filename)
