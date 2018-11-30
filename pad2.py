import os
import numpy as np
import gc

def padX(X):
	padded_X=[]
	dim_size=len(X[0][0][0])
	print dim_size
	required_dim_size=11
	for news in X:
		if len(news)>100:
			news=news[:100]
		for i in range(len(news)):
			sent=news[i]
			if len(sent)>100:
				sent=sent[:100]
			for j in range(len(sent)):
				sent[j]=sent[j][:6]+sent[8:13]
			sent=sent+[[0]*required_dim_size]*(100-len(sent))
			news[i]=sent
		sent=[[0]*required_dim_size]*100
		for _ in range(100-len(news)):
			news.append(sent)
		padded_X.append(np.array(news,np.int8))
	padded_X=np.array(padded_X,np.int8)
	print padded_X.shape
	return padded_X

def processX(X,s):
	for news in X:
		for sent in news:
			for i in range(len(sent)):
				new_sent=[]
				for j in range(13):
					if j==6 or j==7:continue
					new_sent.append(list(s).index(sent[i][j]))
				sent[i]=new_sent
	return X


def run(s):
	names=os.listdir('syntax')

	for name in names:
		X=list(np.load('syntax/'+name)['arr_0'])
		print 'Loaded',name
		X=processX(X,s)
		padded_X=padX(X)
		np.savez_compressed('padded_syntax/'+name,padded_X)
		print 'Done',name


