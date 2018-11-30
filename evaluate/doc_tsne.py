import sys
import numpy as np
from keras.models import load_model, Model
from attention import *
import gc
from sklearn.manifold import TSNE

path=sys.argv[1]
batch_size=32
if len(sys.argv)>2:
    batch_size=int(sys.argv[2])

X=np.load('../train_data_test_X.npz')['arr_0']
required_output_layer_name='attention_layer_1'
model=load_model(path,custom_objects={'AttentiveConcatenateLayer':AttentiveConcatenateLayer,'AttentionLayer':AttentionLayer})
print 'Loaded model'
model.summary()
intermediate_layer_model=Model(inputs=model.input,outputs=model.get_layer(required_output_layer_name).output)
output=intermediate_layer_model.predict(X)
np.savez_compressed('doc_output_A',output)
#intermediate_layer_model.save('intermediate_layer_model')
