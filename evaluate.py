import sys
import numpy as np
from keras.models import load_model

path=sys.argv[1]

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

model=load_model(path)
X=np.load('train_data_test_X.npz')['arr_0']
y_true=np.load('train_data_test_Y.npz')['arr_0']
y_pred=model.predict(X)
print_result(y_true,y_pred)