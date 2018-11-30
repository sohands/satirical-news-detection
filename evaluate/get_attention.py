import sys
import numpy as np
from keras.models import load_model, Model
from attention_2 import AttentionLayer
from attention_3 import AttentiveConcatenateLayer
import gc
from keras.optimizers import SGD
from sklearn.manifold import TSNE

path=sys.argv[1]
batch_size=32
if len(sys.argv)>2:
    batch_size=int(sys.argv[2])

X=np.load('../train_data_test_X.npz')['arr_0']
required_output_layer_name='attentive_concatenate_layer_1'
model=load_model(path,custom_objects={'AttentiveConcatenateLayer':AttentiveConcatenateLayer,'AttentionLayer':AttentionLayer})
print 'Loaded model'
model.summary()
#model.get_layer(required_output_layer_name).call=acl.call
#model.compile(optimizer=SGD(),loss='mse',metrics=[])
intermediate_layer_model=Model(inputs=model.input,outputs=model.get_layer(required_output_layer_name).output)
output=intermediate_layer_model.predict(X,batch_size=batch_size)
np.savez_compressed('attconcat_output_test',output)
# intermediate_layer_model.save('intermediate_layer_model')
