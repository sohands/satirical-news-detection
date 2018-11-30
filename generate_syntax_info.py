from nltk.corpus import sentiwordnet as swn, wordnet as wn
from nltk import pos_tag, ne_chunk
from nltk.chunk import tree2conlltags
import cPickle
import numpy as np
from nltk.data import load
import gc

tagset=load('help/tagsets/upenn_tagset.pickle').keys()

def readObject(filepath):
	f=open(filepath,'rb')
	o=cPickle.load(f)
	f.close()
	return o

def saveObject(o,filepath):
	f=open(filepath,'wb')
	cPickle.dump(o,f,cPickle.HIGHEST_PROTOCOL)
	f.close()

def getWordnetTag(tag):
	if tag[0]=='N':
		return wn.NOUN
	if tag[0]=='J':
		return wn.ADJ
	if tag[0]=='V':
		return wn.VERB
	if tag[0]=='R':
		return wn.ADV
	return None

def getSyntaxInfo(sentence):
	tags=pos_tag(sentence.split())
	ne_tree=ne_chunk(tags)
	ne_tagged=tree2conlltags(ne_tree)
	syntax_info=[]
	caps_range=set(range(ord('A'),ord('Z')+1,1))
	for i in range(len(tags)):
		tag=tags[i]
		ne_tag=ne_tagged[i][2]
		tag_no=tagset.index(tag[1])
		sentiment_score=[0,0,0]
		wordnetTag=getWordnetTag(tag[1])
		if wordnetTag is None:
			synset=wn.synsets(tag[0])
			if len(synset)==0:
				synset=None
			else:
				synset=synset[0]
				sentiSynset=swn.senti_synset(synset.name())
				sentiment_score=[sentiSynset.pos_score(),sentiSynset.neg_score(),sentiSynset.obj_score()]
		else:
			synset=wn.synsets(tag[0],pos=wordnetTag)
			if len(synset)==0:
				synset=None
			else:
				synset=synset[0]
				sentiSynset=swn.senti_synset(synset.name())
				sentiment_score=[sentiSynset.pos_score(),sentiSynset.neg_score(),sentiSynset.obj_score()]
		start_caps=int(ord(tag[0][0]) in caps_range)
		allcaps=1
		for c in tag[0]:
			if ord(c) not in caps_range:
				allcaps=0
				break
		is_number=0
		try:
			n=float(tag[0])
			is_number=1
		except:
			pass
		# for i in range(3):
		# 	sentiment_score[i]=sentiment_score[i]/0.25+4
		iob_tag=ne_tag[0]
		if ne_tag=='O':
			ne_tag=''
		else:
			ne_tag=ne_tag[2:]
		hypernyms=[synset]
		last_two_synsets=[None,None]
		same_synset=[0,0]
		if synset is not None:
			while len(hypernyms[-1].hypernyms())>0:
				hypernyms.append(hypernyms[-1].hypernyms()[0])
			last_two_synsets=[hypernyms[-1].name(),None]
			same_synset[0]=int(last_two_synsets[0]==synset.name())
			if len(hypernyms)>1:
				last_two_synsets[1]=hypernyms[-2].name()
				same_synset[1]=int(last_two_synsets[1]==synset.name())
		syntax_info.append([tag_no]+sentiment_score+[iob_tag,ne_tag]+last_two_synsets+same_synset+[start_caps,allcaps,is_number,len(tag[0])])
	return syntax_info

news_dict=readObject('news_dict_test.pickle')

syntax_X=[]

count=0

for tag in news_dict:
	for news in news_dict[tag]:
		news_X=[]
		for sent in news:
			news_X.append(getSyntaxInfo(sent))
		count+=1
		print count
		syntax_X.append(news_X)
		if count%10000==0:
			np.savez_compressed('syntax_test/syntax_X_'+str(count),syntax_X)
			syntax_X=[]
np.savez_compressed('syntax_test/syntax_X_'+str(count),syntax_X)

# del news_dict

# gc.collect()

# np.savez_compressed('syntax_X',syntax_X)
