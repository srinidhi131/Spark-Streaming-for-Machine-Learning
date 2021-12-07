import pandas as pd
import socket 
import json
import csv
import sys
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from nltk import  sent_tokenize
import string
import matplotlib.pyplot as plt
import sklearn
import pickle
from wordcloud import WordCloud
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,train_test_split,StratifiedKFold,cross_val_score,learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

TCP_IP = "localhost"
TCP_PORT = 6100
s = socket.socket()   
s.connect((TCP_IP, TCP_PORT))


def preprocess(data):
	#batch+=1
	data=data.dropna()
	spam_words=''
	ham_words=''
	# Creating a corpus of spam messages
	for val in data[data['feature2'] == 'spam'].feature1:
		feature1 = val.lower()
		tokens = nltk.word_tokenize(feature1)
	for words in tokens:
		spam_words = spam_words + words + ' '
	# Creating a corpus of ham messages
	for val in data[data['feature2'] == 'ham'].feature1:
		feature1 = feature1.lower()
		tokens = nltk.word_tokenize(feature1)
	for words in tokens:
		ham_words = ham_words + words + ' '
	spam_wordcloud = WordCloud(width=500, height=300).generate(spam_words)
	ham_wordcloud = WordCloud(width=500, height=300).generate(ham_words)
	#Spam Word cloud
	plt.figure( figsize=(10,8), facecolor='w')
	plt.imshow(spam_wordcloud)
	plt.axis("off")
	plt.tight_layout(pad=0)
	plt.show()
	#Creating Ham wordcloud
	plt.figure( figsize=(10,8), facecolor='g')
	plt.imshow(ham_wordcloud)
	plt.axis("off")
	plt.tight_layout(pad=0)
	plt.show()
	data = data.replace(['ham','spam'],[0, 1])
	#remove the punctuations and stopwords
	import string
	def text_process(text): 
		text = text.translate(str.maketrans('', '', string.punctuation))
		text = [word for word in text.split() if word.lower() not in stopwords.words('english')]    
		return " ".join(text)
	data['feature1'] = data['feature1'].apply(text_process)
	text = pd.DataFrame(data['feature1'])
	label = pd.DataFrame(data['feature2'])
	from collections import Counter
	total_counts = Counter()
	for i in range(len(text)):
		for word in text.values[i][0].split(" "):
			total_counts[word] += 1
	vocab = sorted(total_counts, key=total_counts.get, reverse=True)
	vocab_size = len(vocab)
	word2idx = {}
	#print vocab_size
	for i, word in enumerate(vocab):
		word2idx[word] = i
	# Text to Vector
	def text_to_vector(text):
		word_vector = np.zeros(vocab_size)
		for word in text.split(" "):
			if word2idx.get(word) is None:
				continue
			else:
				word_vector[word2idx.get(word)] += 1
		return np.array(word_vector)
	# Convert all titles to vectors
	word_vectors = np.zeros((len(text), len(vocab)), dtype=np.int_)
	for i, (_, text_) in enumerate(text.iterrows()):
		word_vectors[i] = text_to_vector(text_[0])
	features=word_vectors
	X_train, X_test, y_train, y_test = train_test_split(features, data['feature2'], test_size=0.01)
	return X_train, y_train
def recieve_data(TCP_IP,TCP_PORT):
	d=""
	u=""
	temp=""
	batch=0
	batchsize=sys.maxsize
	check=-sys.maxsize
	sgd = SGDClassifier(loss='log')
	perc = Perceptron()
	mnb = MultinomialNB(alpha=0.2)
	pac = PassiveAggressiveClassifier(C=0.5)
	clfs = {'SGD' : sgd, 'perc' : perc, 'NB': mnb, 'PAC' : pac}
	test_train = 1
	# recieve data from the server and decoding to get the string.
	while(True):
		d=s.recv(1024).decode()
		length = len(d)
		if(length==0):
			break
		temp=d.split("\n",1)
		u+=temp[0]
		if len(temp)==2 and test_train>batch:
			batch+=1
			var=json.loads(u)
			fi=list(var.values())
			df=pd.DataFrame.from_dict(fi)	
			X_train, y_train=preprocess(df)
			#print(X_train.shape)
			
			if batch==1:
				batchsize=df.shape[0]
				test_train=int(30345/batchsize)
				#print(batchsize)
				check=X_train.shape[1]
			if check>X_train.shape[1]:
				tempi=np.zeros((X_train.shape[0],check))
				tempi[:X_train.shape[0],:X_train.shape[1]]=X_train
				X_train=tempi.astype(int)
				#print(X_train.shape)
			if check<X_train.shape[1]:
				X_train=X_train[:,:check]
			for k,clf in clfs.items():
				#print(X_train.shape)
				clf.partial_fit(X_train, y_train,classes=np.unique(y_train))
			u=temp[1]
		
			
	s.close()
	#     return
recieve_data(TCP_IP,TCP_PORT)
