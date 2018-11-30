from keras.models import Sequential
import numpy as np
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from auxillary import F1

model=Sequential()
model.add(Dense(150,activation='relu',input_shape=(300,)))
#model.add(Dropout(0.3))
model.add(Dense(75))
model.add(Dense(2,activation='softmax'))

model.compile(optimizer=SGD(0.05,nesterov=True,momentum=0.9),loss='categorical_crossentropy',metrics=['acc'])

X=np.load('last_sentences_train.npz')['arr_0']
Y=np.load('last_sentences_train_Y.npz')['arr_0']

model.fit(X,Y,epochs=10)
print F1(Y,model.predict(X))
