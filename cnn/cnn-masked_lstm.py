import numpy as np
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Merge, Dropout, Embedding, Reshape, LSTM, Permute, Masking, RepeatVector
from keras.optimizers import SGD, RMSprop
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling1D
from keras.layers.merge import multiply
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.callbacks import ModelCheckpoint
import cPickle
from auxillary import *
from attention import *

window_size=3
max_sent_length=100
word_embedding_size=300
required_sent_embedding_size=300
max_news_length=100
input_shape=(max_news_length,300,max_sent_length,1)
learning_rate=0.05
length_embedding_words=299527
lstm_num_hidden=150
# DATA_FOLDER='/media/sohan/Windows/Users/Sohan/Downloads/'
DATA_FOLDER='../'
trainX,trainY=loadTrainData(DATA_FOLDER+'train_data_X.npz',DATA_FOLDER+'train_data_Y.npz',provideMask=True)
valX,valY=loadTrainData(DATA_FOLDER+'train_data_dev_X.npz',DATA_FOLDER+'train_data_dev_Y.npz',provideMask=True)
valdata=(valX,valY)
embedding_matrix=readObject('../embedding_matrix.pickle')
save_filename='models/cnn_masked_lstm_1'

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

model=Sequential()
model.add(Reshape((max_news_length*max_sent_length,),input_shape=(max_news_length,max_sent_length,1)))
model.add(Embedding(length_embedding_words,word_embedding_size,weights=[embedding_matrix],
	input_length=max_news_length*max_sent_length,trainable=False))
model.add(Reshape((max_news_length,max_sent_length,word_embedding_size)))
model.add(TimeDistributed(Conv1D(filters=required_sent_embedding_size,kernel_size=3)))
model.add(TimeDistributed(MaxPooling1D(98)))
model.add(TimeDistributed(Flatten()))

mask=Sequential()
mask.add(Flatten(input_shape=(max_news_length,1)))
mask.add(RepeatVector(required_sent_embedding_size))
mask.add(Permute((2,1)))

masked_model=Sequential()
masked_model.add(Merge([model,mask],mode='mul'))
masked_model.add(Masking(mask_value=0))
masked_model.add(Bidirectional(LSTM(lstm_num_hidden,return_sequences=False)))
#masked_model.add(AttentionLayer())
masked_model.add(Dropout(0.3))
masked_model.add(Dense(100,activation='relu'))
masked_model.add(Dropout(0.3))
masked_model.add(Dense(2,activation='softmax'))
masked_model.summary()

#optimizer=SGD(learning_rate,decay=1e-6,momentum=0.9,nesterov=True,clipnorm=1.)
optimizer=RMSprop(learning_rate)
metrics=['accuracy','categorical_crossentropy']
loss='categorical_crossentropy'
checkpoint=ModelCheckpoint('bestmodels/cnn_mblstm',monitor='val_acc',save_best_only=True,mode='max',verbose=1)

masked_model.compile(optimizer=optimizer,loss=loss,metrics=metrics)

masked_model.fit(trainX,trainY,epochs=20,batch_size=32,validation_data=valdata,callbacks=[checkpoint])

masked_model.save(save_filename)
