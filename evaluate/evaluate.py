import sys
import numpy as np
from keras.models import load_model
from attention_2 import *
import gc

path=sys.argv[1]
batch_size=32
if len(sys.argv)>2:
    batch_size=int(sys.argv[2])

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
    print 'Model path:',path
    print 'Precision:',int(10000*precision)/100.
    print 'Recall:',int(10000*recall)/100.
    print 'F-score:',int(10000*f1)/100.
    print 'Accuracy:',int(10000*accuracy)/100.

print AttentiveConcatenateLayer

model=load_model(path,custom_objects={'AttentiveConcatenateLayer':AttentiveConcatenateLayer,'AttentionLayer':AttentionLayer})
print 'Loaded model'
print 'Results on Validation data'
X=np.load('../train_data_dev_X.npz')['arr_0']
y_true=np.load('../train_data_dev_Y.npz')['arr_0']
y_pred=model.predict(X,batch_size=batch_size)
print_result(y_true,y_pred)
del X
del y_pred
del y_true
gc.collect()
print 'Results on Test data'
X=np.load('../train_data_test_X.npz')['arr_0']
y_true=np.load('../train_data_test_Y.npz')['arr_0']
y_pred=model.predict(X,batch_size=batch_size)
print_result(y_true,y_pred)
