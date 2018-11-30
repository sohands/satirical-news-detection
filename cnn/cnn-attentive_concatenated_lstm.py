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
from attention import *

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
testX=np.load(DATA_FOLDER+'train_data_test_X.npz')['arr_0']
testY=np.load(DATA_FOLDER+'train_data_test_Y.npz')['arr_0']
valdata=(valX,valY)
embedding_matrix=readObject('../embedding_matrix.pickle')
save_filename='models/cnn_dac_dbgru_da_te'

def print_result(y_true,y_pred):
    y_true=np.argmax(y_true,-1)
    y_pred=np.argmax(y_pred,-1)
    c1=np.sum(y_true*y_pred)
    c2=np.sum(y_pred)
    c3=np.sum(y_true)
    precision=float(c1)/c2
    recall=float(c1)/c3
    f1=(2*precision*recall)/(precision+recall)
    accuracy=float(np.sum(np.equal(y_true,y_pred)))/len(y_true)
    print 'Precision:',int(10000*precision)/100.
    print 'Recall:',int(10000*recall)/100.
    print 'F-score:',int(10000*f1)/100.
    print 'Accuracy:',int(10000*accuracy)/100.

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
	input_length=max_news_length*max_sent_length,trainable=True))
model.add(Dense(300,activation='relu'))
model.add(Dropout(0.2))
model.add(Reshape((max_news_length,max_sent_length,word_embedding_size)))
model.add(TimeDistributed(Conv1D(filters=required_sent_embedding_size,kernel_size=3)))
model.add(TimeDistributed(MaxPooling1D(98)))
model.add(TimeDistributed(Flatten()))
#model.add(Dropout(0.2))
model.add(AttentiveConcatenateLayer(length=max_news_length,hidden_layers=(75,)))
model.add(Bidirectional(LSTM(lstm_num_hidden,return_sequences=True)))
#model.add(Bidirectional(GRU(lstm_num_hidden,return_sequences=True)))
#model.add(Bidirectional(GRU(lstm_num_hidden,return_sequences=True)))
model.add(AttentionLayer(hidden_layers=(75,)))
model.add(Dropout(0.5))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))
model.summary()

optimizer=SGD(learning_rate,decay=1e-6,momentum=0.9,nesterov=True,clipnorm=1.)
metrics=['accuracy','categorical_crossentropy']
loss='categorical_crossentropy'
checkpoint=ModelCheckpoint('bestmodels/cnn_dac_dbgru_da_te',monitor='val_acc',save_best_only=True,mode='max',verbose=1)

model.compile(optimizer=optimizer,loss=loss,metrics=metrics)

max_f1=0

for i in range(20):
	model.fit(trainX,trainY,epochs=1,batch_size=24)
	# model.save('epochmodels/cnn_dac_dbgru_da_te_'+str(i))
	#val_f1=F1(valY,model.predict(valX))
	#print 'Epoch',i,': Validation F1 score:',val_f1
	print 'Epoch',i
        print 'Validation'
        print_result(valY,model.predict(valX))
        print 'Test'
	print_result(testY,model.predict(testX))
	print ''
	continue
	if val_f1>max_f1:
		max_f1=val_f1
		model.save('bestmodels/cnn_dac_dbgru_da_te_f1')
		print 'Saved model'
	else:
		print ''
# model.fit(trainX,trainY,epochs=20,batch_size=32,validation_data=valdata,callbacks=[checkpoint])

# model.save(save_filename)
