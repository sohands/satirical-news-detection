#import sys
import numpy as np
#from keras.models import Model, load_model
#from attention_2 import *
import cPickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def readObject(filename):
	f=open(filename,'rb')
	o=cPickle.load(f)
	f.close()
	return o

def saveObject(o,filename):
	f=open(filename,'wb')
	cPickle.dump(o,f,cPickle.HIGHEST_PROTOCOL)
	f.close()

def reduce_dims(X):
	return PCA(n_components=2).fit_transform(X)

#path=sys.argv[1]
#batch_size=32
#if len(sys.argv)>2:
#	batch_size=int(sys.argv[2])
#required_output_layer_name='dense_1'


#model=load_model(path,custom_objects={'AttentiveConcatenateLayer':AttentiveConcatenateLayer,'AttentionLayer':AttentionLayer})
#print 'Loaded model'
#embedding_matrix=readObject('../embedding_matrix.pickle')
#print 'Loaded embedding matrix'
#dimsize=len(embedding_matrix[0])
#padding_required=[]
#for i in range((100-len(embedding_matrix)%100)%100):
#	padding_required.append([0]*dimsize)
#embedding_matrix=np.concatenate([embedding_matrix,padding_required],axis=0)
#X=embedding_matrix.reshape((-1,100,100,1))
#print X.shape
#intermediate_layer_model=Model(inputs=model.input,outputs=model.get_layer(required_output_layer_name).output)
#print 'Prepared intermediate model'
#output=intermediate_layer_model.predict(X,batch_size=batch_size)
#print output.shape
#np.savez_compressed('word_embeddings',output)
#print 'Saved output'
output=np.load('word_embeddings.npz')['arr_0']
print output.shape
output=output.reshape((-1,300))
embedding_matrix=np.load('../embedding_matrix.pickle')
X=np.concatenate([output,embedding_matrix],axis=0)
tsneX=reduce_dims(X)
print tsneX.shape
# saveObject(tsneX,'tsne_words.pickle')
embedding_words=readObject('../embedding_words.pickle')
words=embedding_words+['hcsvhjvajs']*(len(output)-len(embedding_words))
for word in embedding_words:
	words.append('glove_'+word)
print len(words),tsneX.shape
d={}
for i in range(len(words)):
	d[words[i]]=tsneX[i]
saveObject(d,'d.pickle')