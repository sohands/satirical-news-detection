import cPickle
import numpy as np
import sys
from copy import deepcopy
import gc
import os

def readObject(filepath):
	f=open(filepath,'rb')
	o=cPickle.load(f)
	f.close()
	return o

def saveObject(o,filepath):
	f=open(filepath,'wb')
	cPickle.dump(o,f,cPickle.HIGHEST_PROTOCOL)
	f.close()

def loadTempFiles(temp_filenames):
	X=[]
	Y=[]
	for filename in temp_filenames:
		(x,y)=readObject(filename)
		# os.system('rm '+temp_filenames)
		X+=x
		Y+=y
		print 'Loaded',filename
	return X,Y

def getTrainData(newsFile='../news_dict.pickle',max_sent_length=100,max_news_length=100):

	def isNumber(i):
		try:
			i=float(i)
			return True
		except:
			return False

	X=[]
	Y=[]

	news_dict=readObject(newsFile)

	embedding_words=readObject('../embedding_words.pickle')

	embedding_words_set=set(embedding_words)

	print len(embedding_words_set)

	all_words=[]

	for tag in news_dict:
		for news in news_dict[tag]:
			for sent in news:
				all_words+=sent.split()
		all_words=list(set(all_words))

	print len(all_words)
	embedding_word_index_dict={}

	for i in range(len(embedding_words)):
		embedding_word_index_dict[embedding_words[i]]=i

	unk_index=embedding_words.index('unk')
	number_index=embedding_words.index('#')
	pad_index=embedding_words.index('xpadx')

	word_index_dict={'xpadx':pad_index,'unk':unk_index,'#':number_index}
	for word in all_words:
		index=unk_index
		if word.lower()=='unk':
			index=unk_index
		elif word=='xpadx':
			index=pad_index
		elif word in embedding_words_set:
			index=embedding_word_index_dict[word]
		elif word.lower() in embedding_words_set:
			index=embedding_word_index_dict[word.lower()]
		elif isNumber(word):
			index=number_index
		word_index_dict[word]=index

	del embedding_words
	del embedding_words_set
	del all_words
	del embedding_word_index_dict

	gc.collect()

	total=len(news_dict[True])+len(news_dict[False])
	print total
	count=0
	print 'Starting...'
	temp_filenames=[]

	for tag in news_dict:
		for news in news_dict[tag]:
			# if count<100000:
			# 	count+=1
			# 	continue
			if len(news)>max_news_length:
				news=news[:max_news_length]
			# elif len(news)<max_news_length:
			# 	news=deepcopy(news)+['xpadx']*(max_news_length-len(news))
			news_sents=[]
			for sent in news+['xpadx']*(max_news_length-len(news)):
				word_indices=[]
				words=sent.split()
				if len(words)>max_sent_length:
					words=words[:max_sent_length]
				elif len(words)<max_sent_length:
					words+=['xpadx']*(max_sent_length-len(words))
				for word in words:
					index=word_index_dict[word]
					# index=unk_index
					# if word.lower()=='unk':
					# 	index=unk_index
					# elif word=='xpadx':
					# 	index=pad_index
					# elif word in embedding_words_set:
					# 	index=word_index_dict[word]
					# elif word.lower() in embedding_words_set:
					# 	index=word_index_dict[word.lower()]
					# elif isNumber(word):
					# 	index=number_index
					word_indices.append([index])
					# print 'Found',word,word_index_dict[word]
					# if not multiple:
					# 	print 'Popping',word
					# 	word_index_dict.pop(word)
				news_sents.append(word_indices)
			X.append(np.array(news_sents,dtype=np.int32))
			if tag:
				Y.append([1,0])
			else:
				Y.append([0,1])
			count+=1
			if count%100==0:
				# gc.collect()
				print 'Completed:',count
			if count%50000==0:
				saveObject((X,Y),'tempfile'+str(count))
				temp_filenames.append('tempfile'+str(count))
				X=[]
				Y=[]
				gc.collect()

	del word_index_dict

	gc.collect()

	# temp_filenames=['tempfile50000','tempfile100000']

	tempX,tempY=loadTempFiles(temp_filenames)
	X=tempX+X
	Y=tempY+Y
	X=np.array(X)
	Y=np.array(Y)
	# print ''
	print X.shape,Y.shape

	saveObject((X,Y),'train_data_test.pickle')

	return X,Y

def getMask(X):

	#embedding_words=readObject('../embedding_words.pickle')
	pad_index=45250#embedding_words.index('xpadx')
	mask=np.ndarray.astype(np.all(np.not_equal(X,pad_index),axis=2),np.float32)
	return mask

def loadTrainData(X_filename='../train_data_X.npz',Y_filename='../train_data_Y.npz',npz=True,provideMask=False):
	if provideMask is False:
		if npz:
			X=np.load(X_filename)['arr_0']
			Y=np.load(Y_filename)['arr_0']
			return X,Y
		else:
			(X,Y)=readObject(X_filename)
			return X,Y
	else:
		if npz:
			X=np.load(X_filename)['arr_0']
			Y=np.load(Y_filename)['arr_0']
		else:
			(X,Y)=readObject(X_filename)
		return [X,getMask(X)],Y

def F1(y_true,y_pred):
	y_true=np.argmax(y_true,-1)
	y_pred=np.argmax(y_pred,-1)
	c1=np.sum(y_true*y_pred)
	c2=np.sum(y_pred)
	c3=np.sum(y_true)
	precision=float(c1)/c2
	recall=float(c1)/c3
	f1=(2*precision*recall)/(precision+recall)
	return int(10000*f1)/100.

# getTrainData()
