import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling1D, Flatten, Dropout, Merge, Embedding, Reshape
from keras.optimizers import SGD, Adadelta
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling1D
from keras.layers.merge import average
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint
import cPickle
from auxillary import *

window_size=2
max_sent_length=100
word_embedding_size=600
required_sent_embedding_size=300
max_news_length=100
input_shape=(max_news_length,300,max_sent_length,1)
learning_rate=0.05
length_embedding_words=299527
# DATA_FOLDER='/media/sohan/Windows/Users/Sohan/Downloads/'
DATA_FOLDER='../'
trainX,trainY=loadTrainData(DATA_FOLDER+'train_data_X.npz',DATA_FOLDER+'train_data_Y.npz')
valX,valY=loadTrainData(DATA_FOLDER+'train_data_dev_X.npz',DATA_FOLDER+'train_data_dev_Y.npz')
valdata=(valX,valY)
embedding_matrix=readObject('../embedding_matrix.pickle')
embedding_matrix_fastext=readObject('../embedding_matrix_fastext.pickle')
embedding_matrix=np.concatenate([embedding_matrix,embedding_matrix_fastext],axis=-1)
save_filename='models/deep_cnn_cnn_1'

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
#model.add(TimeDistributed(Dense(300,activation='linear')))
#model.add(Dropout(0.1))
model.add(Reshape((max_news_length,max_sent_length,word_embedding_size/2,2)))
model.add(TimeDistributed(Conv2D(filters=required_sent_embedding_size/2,kernel_size=(3,word_embedding_size/2),strides=(window_size,1))))
model.summary()
model.add(TimeDistributed(Conv2D()))
#model.add(TimeDistributed(Conv1D(filters=required_sent_embedding_size/2,kernel_size=3)))
#model.add(TimeDistributed(Conv1D(filters=required_sent_embedding_size,kernel_size=3)))
model.add(Reshape((max_news_length,max_sent_length,)))
model.add(TimeDistributed(MaxPooling1D(96)))
model.add(TimeDistributed(Flatten()))
#model.add(Dropout(0.3))
model.add(Conv1D(filters=150,kernel_size=3))
#model.add(Conv1D(filters=300,kernel_size=3))
model.add(MaxPooling1D(98))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2,activation='softmax'))
model.summary()

optimizer=SGD(learning_rate,decay=1e-6,momentum=0.9,nesterov=True,clipnorm=1.)
metrics=['accuracy','categorical_crossentropy']
loss='categorical_crossentropy'
checkpoint=ModelCheckpoint('bestmodels/deep_cnn_cnn',monitor='val_acc',save_best_only=True,mode='max',verbose=1)

model.compile(optimizer=optimizer,loss=loss,metrics=metrics)

model.fit(trainX,trainY,epochs=20,batch_size=32,validation_data=valdata,callbacks=[checkpoint])

model.save(save_filename)
