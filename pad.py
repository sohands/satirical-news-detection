import os
import numpy as np
import gc
import cPickle

def readObject(filepath):
	f=open(filepath,'rb')
	o=cPickle.load(f)
	f.close()
	return o

def saveObject(o,filepath):
	f=open(filepath,'wb')
	cPickle.dump(o,f,cPickle.HIGHEST_PROTOCOL)
	f.close()

def padX(X):
	padded_X=[]
	# print X[0][0][0]
	dim_size=len(X[0][0][0])
	# print dim_size
	required_dim_size=11
	for news in X:
		if len(news)>100:
			news=news[:100]
		for i in range(len(news)):
			sent=news[i]
			if len(sent)>100:
				sent=sent[:100]
			# for j in range(len(sent)):
			# 	sent[j]=sent[j][:6]+sent[j][8:13]
			sent=sent+[[0]*required_dim_size]*(100-len(sent))
			news[i]=sent
		sent=[[0]*required_dim_size]*100
		for _ in range(100-len(news)):
			news.append(sent)
		# print 'News',len(news),type(news),len(news[0]),len(news[1]),type(news[0]),type(news[0][0]),len(news[0][0]),news[0][0]
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
					if sent[i][j] not in s[j]:
						new_sent.append(0)
						continue
					new_sent.append(s[j].index(sent[i][j]))
				sent[i]=new_sent
	return X

def generateS():
	names=os.listdir('syntax')
	s=[set() for _ in range(14)]
	for name in names:
		X=list(np.load('syntax/'+name)['arr_0'])
		for news in X:
			for sent in news:
				for w in sent:
					for j in range(13):
						if j==6 or j==7:
							continue
						s[j].add(w[j])
		print 'Processed s:',name
	for i in range(len(s)):
		s[i]=list(s[i])
	return s

def run(s):
	names=os.listdir('syntax_test')

	for name in names:
		X=list(np.load('syntax_test/'+name)['arr_0'])
		print 'Loaded',name
		X=list(processX(X,s))
		print 'Processed X'
		padded_X=padX(X)
		np.savez_compressed('padded_syntax_test/'+name,padded_X)
		print 'Done',name

# s=generateS()
# saveObject(s,'s.pickle')
run(readObject('s.pickle'))
