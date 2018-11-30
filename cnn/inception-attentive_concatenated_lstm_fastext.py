import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Merge, Embedding, Reshape, Permute, LSTM, GRU
from keras.optimizers import SGD, Adadelta
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling1D
from keras.layers import concatenate
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.callbacks import ModelCheckpoint
import cPickle
from auxillary import *
from attention import *

window_size=3
max_sent_length=100
word_embedding_size=300
required_sent_embedding_size=150
max_news_length=100
input_shape=(max_news_length,300,max_sent_length,1)
learning_rate=0.005
length_embedding_words=299527
lstm_num_hidden=150
# DATA_FOLDER='/media/sohan/Windows/Users/Sohan/Downloads/'
DATA_FOLDER='../'
trainX,trainY=loadTrainData(DATA_FOLDER+'train_data_X.npz',DATA_FOLDER+'train_data_Y.npz')
valX,valY=loadTrainData(DATA_FOLDER+'train_data_dev_X.npz',DATA_FOLDER+'train_data_dev_Y.npz')
valdata=(valX,valY)
embedding_matrix=readObject('../embedding_matrix.pickle')
# embedding_matrix_fastext=readObject('../embedding_matrix_fastext.pickle')
# embedding_matrix=np.concatenate([embedding_matrix,embedding_matrix_fastext],axis=-1)
save_filename='models/cnn_dac_dbgru_da_te'
#print embedding_matrix.shape
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

original_inputs=Input(shape=(max_news_length,max_sent_length,1))
inputs=Reshape((max_news_length*max_sent_length,))(original_inputs)

embeddings=Embedding(length_embedding_words,word_embedding_size,weights=[embedding_matrix],
	input_length=max_news_length*max_sent_length,trainable=True)(inputs)
embeddings=Dense(150,activation='relu')(embeddings)
embeddings=Dropout(0.1)(embeddings)
embeddings=Reshape((max_news_length,max_sent_length,word_embedding_size/2))(embeddings)

conv1x1=TimeDistributed(Dense(required_sent_embedding_size/2,activation='relu'))(embeddings)
# conv1x1=TimeDistributed(Reshape((max_news_length,max_sent_length,required_sent_embedding_size)))(conv1x1)
conv1x1=TimeDistributed(MaxPooling1D(100))(conv1x1)
conv1x1=TimeDistributed(Flatten())(conv1x1)
#conv1x1=Permute((1,3,2))(conv1x1)

conv3x1=TimeDistributed(Dense(required_sent_embedding_size/2,activation='relu'))(embeddings)
# conv3x1=TimeDistributed(Reshape((max_news_length,max_sent_length,required_sent_embedding_size)))(conv3x1)
conv3x1=TimeDistributed(Conv1D(filters=required_sent_embedding_size,kernel_size=3,activation='relu'))(conv3x1)
conv3x1=TimeDistributed(MaxPooling1D(98))(conv3x1)
conv3x1=TimeDistributed(Flatten())(conv3x1)
#conv3x1=Permute((1,3,2))(conv3x1)

conv5x1=TimeDistributed(Dense(required_sent_embedding_size/2,activation='relu'))(embeddings)
# conv5x1=TimeDistributed(Reshape((max_news_length,max_sent_length,required_sent_embedding_size)))(conv5x1)
conv5x1=TimeDistributed(Conv1D(filters=required_sent_embedding_size,kernel_size=5,activation='relu'))(conv5x1)
conv5x1=TimeDistributed(MaxPooling1D(96))(conv5x1)
conv5x1=TimeDistributed(Flatten())(conv5x1)
#conv5x1=Permute((1,3,2))(conv5x1)

merged_conv=concatenate([conv1x1,conv3x1,conv5x1],axis=-1)
#merged_conv=TimeDistributed(Conv1D(filters=required_sent_embedding_size,kernel_size=3,activation='relu'))(merged_conv)
#merged_conv=TimeDistributed(MaxPooling1D(298))(merged_conv)
merged_conv=TimeDistributed(Dense(required_sent_embedding_size,activation='relu'))(merged_conv)
#merged_conv=TimeDistributed(Flatten())(merged_conv)
#merged_conv=Dropout(0.3)(merged_conv)

# model=Model(inputs=original_inputs,outputs=merged_conv)

# model=Sequential()
# model.add(merged_conv)
# model.add(AttentiveConcatenateLayer(length=max_news_length,hidden_layers=(75,)))
# model.add(Bidirectional(GRU(lstm_num_hidden,return_sequences=True)))
# model.add(Bidirectional(GRU(lstm_num_hidden,return_sequences=True)))
# model.add(Bidirectional(GRU(lstm_num_hidden,return_sequences=True)))
# model.add(AttentionLayer(hidden_layers=(75,)))
# model.add(Dropout(0.5))
# model.add(Dense(100,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2,activation='softmax'))
# model.summary()

dac=AttentiveConcatenateLayer(length=max_news_length,hidden_layers=(75,))(merged_conv)
dbgru=Bidirectional(GRU(lstm_num_hidden,return_sequences=True))(dac)
dbgru=Bidirectional(GRU(lstm_num_hidden,return_sequences=True))(dbgru)
dbgru=Bidirectional(GRU(lstm_num_hidden,return_sequences=True))(dbgru)
da=AttentionLayer(hidden_layers=(75,))(dbgru)

FC=Dropout(0.3)(da)
FC=Dense(100,activation='relu')(FC)
FC=Dropout(0.3)(FC)
predictions=Dense(2,activation='softmax')(FC)

model=Model(inputs=original_inputs,outputs=predictions)

optimizer=SGD(learning_rate,decay=1e-6,momentum=0.9,nesterov=True,clipnorm=1.)
metrics=['accuracy','categorical_crossentropy']
loss='categorical_crossentropy'
checkpoint=ModelCheckpoint('bestmodels/cnn_dac_dbgru_da_te',monitor='val_acc',save_best_only=True,mode='max',verbose=1)

model.compile(optimizer=optimizer,loss=loss,metrics=metrics)

max_f1=0

for i in range(20):
	#for j in range(0,110000,10000):
	model.fit(trainX,trainY,epochs=1,batch_size=16)
	# model.save('epochmodels/cnn_dac_dbgru_da_te_'+str(i))
	val_f1=F1(valY,model.predict(valX))
	print 'Epoch',i,': Validation F1 score:',val_f1,
	if val_f1>max_f1:
		max_f1=val_f1
		model.save('bestmodels/ca-inception_dac_dbgru_da_te_trem_f1')
		print 'Saved model'
	else:
		print ''
# model.fit(trainX,trainY,epochs=20,batch_size=32,validation_data=valdata,callbacks=[checkpoint])

# model.save(save_filename)
