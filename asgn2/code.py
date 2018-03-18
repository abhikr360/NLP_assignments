###### IMPORTS ########

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file
from gensim.models.keyedvectors import KeyedVectors
from scipy.sparse import hstack
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from gensim.models import doc2vec

#### LOAD DATA #####

traindatafile = "../asgn2data/aclImdb/train/labeledBow_shuffled.feat"
tr_data = load_svmlight_file(traindatafile)
Xtr = tr_data[0];
Ytr = tr_data[1];


testdatafile = "../asgn2data/aclImdb/test/labeledBow.feat"
ts_data = load_svmlight_file(testdatafile)
Xts = ts_data[0];
temp=csr_matrix((25000, 4))
Xts = hstack([Xts,temp])
Xts = Xts.tocsr()
Yts = ts_data[1];









##### DOCUMENT REPRESENTATIONS #######

# Binary bag of words
def getBinaryBagOfWords():
	xtr = (Xtr !=0)
	xts = (Xts !=0)
	return xtr, xts


# Normalized Term frequency
def getNormalizedTFReprsentation():
	xtr = normalize(Xtr, norm='l1', axis=1)
	xts = normalize(Xts, norm='l1', axis=1)
	return xtr, xts



# TFIDF
def getTFIDF():
	tfidf_transformer = TfidfTransformer()
	xtr = tfidf_transformer.fit_transform(Xtr)
	xts = tfidf_transformer.fit_transform(Xts)
	return xtr, xts

# Word2Vec Avg 
def getWord2VecAvg():
	vocab = [line.rstrip('\n') for line in open('../asgn2data/aclImdb/imdb.vocab')]
	Word2Vec_word_vectors = KeyedVectors.load_word2vec_format('../asgn2data/word2vec.bin', binary=True)
	ret1=np.zeros((25000, 300))
	# ret2=np.zeros((25000, 300))
	
	for i in range(Xtr.shape[0]):
		k=0
		temp = Xtr.getrow(i)
		temp = temp.toarray()
		for j in range(Xtr.shape[1]):
			if(temp[0][j]):
				if(vocab[j] in Word2Vec_word_vectors.vocab):
					ret1[i] += Word2Vec_word_vectors[vocab[j]]
					k+=1
		ret1[i] = ret1[i]/k
		print "  Iteration %d out of %d\r" % (i,Xtr.shape[0]) ,

	print("")
	print('Train features prepared')
	# for i in range(Xts.shape[0]):
	# 	k=0
	# 	temp = Xts.getrow(i)
	# 	temp = temp.toarray()
	# 	for j in range(Xts.shape[1]):
	# 		if(temp[0][j]):
	# 			if(vocab[j] in Word2Vec_word_vectors.vocab):
	# 				ret2[i] += Word2Vec_word_vectors[vocab[j]]
	# 				k+=1
	# 	ret2[i] = ret2[i]/k
	# 	print "  Iteration %d out of %d\r" % (i,Xts.shape[0]) ,
	print("")

	np.save('train_simple_word2vec_shuffled.npy', ret1)
	# np.save('test_simple_word2vec.npy', ret2)

	# ret1 = np.load('train_simple_word2vec.npy')
	ret2 = np.load('test_simple_word2vec.npy')
	return ret1, ret2


# Word2Vec Weighted Avg with tfidf 
def getWord2VecweightedAvg():
	vocab = [line.rstrip('\n') for line in open('../asgn2data/aclImdb/imdb.vocab')]
	Word2Vec_word_vectors = KeyedVectors.load_word2vec_format('../asgn2data/word2vec.bin', binary=True)
	ret1=np.zeros((25000, 300))
	# ret2=np.zeros((25000, 300))

	tfidf1, tfidf2 = getTFIDF()
	
	for i in range(Xtr.shape[0]):
		k=0
		temp = Xtr.getrow(i)
		temp = temp.toarray()
		t = tfidf1.getrow(i)
		t = t.toarray()
		for j in range(Xtr.shape[1]):
			if(temp[0][j]):
				if(vocab[j] in Word2Vec_word_vectors.vocab):
					ret1[i] += t[0][j]*Word2Vec_word_vectors[vocab[j]]
					k+=t[0][j]
		ret1[i] = ret1[i]/k
		print "  Iteration %d out of %d\r" % (i,Xtr.shape[0]) ,

	print("")
	print('Train features prepared')
	# for i in range(Xts.shape[0]):
	# 	k=0
	# 	temp = Xts.getrow(i)
	# 	temp = temp.toarray()
	# 	t = tfidf2.getrow(i)
	# 	t = t.toarray()
	# 	for j in range(Xts.shape[1]):
	# 		if(temp[0][j]):
	# 			if(vocab[j] in Word2Vec_word_vectors.vocab):
	# 				ret2[i] += t[0][j]*Word2Vec_word_vectors[vocab[j]]
	# 				k+=t[0][j]
	# 	ret2[i] = ret2[i]/k
	# 	print "  Iteration %d out of %d\r" % (i,Xts.shape[0]) ,
	# print("")

	np.save('train_weighted_word2vec_shuffled.npy', ret1)
	# np.save('test_weighted_word2vec.npy', ret2)

	# ret1 = np.load('train_weighted_word2vec.npy')
	ret2 = np.load('test_weighted_word2vec.npy')
	return ret1, ret2


# Glove Avg
def getGloveAvg():
	vocab = [line.rstrip('\n') for line in open('../asgn2data/aclImdb/imdb.vocab')]
	Glove_vectors = KeyedVectors.load_word2vec_format('../asgn2data/Glove.6B.300d.txt', binary=False)
	ret1=np.zeros((25000, 300))
	# ret2=np.zeros((25000, 300))
	
	for i in range(Xtr.shape[0]):
		k=0
		temp = Xtr.getrow(i)
		temp = temp.toarray()
		for j in range(Xtr.shape[1]):
			if(temp[0][j]):
				if(vocab[j] in Glove_vectors.vocab):
					ret1[i] += Glove_vectors[vocab[j]]
					k+=1
		ret1[i] = ret1[i]/k
		print "  Iteration %d out of %d\r" % (i,Xtr.shape[0]) ,

	print("")
	print('Train features prepared')

	# for i in range(Xts.shape[0]):
	# 	k=0
	# 	temp = Xts.getrow(i)
	# 	temp = temp.toarray()
	# 	for j in range(Xts.shape[1]):
	# 		if(temp[0][j]):
	# 			if(vocab[j] in Glove_vectors.vocab):
	# 				ret2[i] += Glove_vectors[vocab[j]]
	# 				k+=1
	# 	ret2[i] = ret2[i]/k
	# 	print "  Iteration %d out of %d\r" % (i,Xts.shape[0]) ,
	# print("")

	np.save('train_simple_glove_shuffled.npy', ret1)
	# np.save('test_simple_glove.npy', ret2)

	# ret1 = np.load('train_simple_glove.npy')
	ret2 = np.load('test_simple_glove.npy')

	return ret1, ret2

# Glove Weighted Avg with tfidf 
def getGloveWeightedAvg():
	vocab = [line.rstrip('\n') for line in open('../asgn2data/aclImdb/imdb.vocab')]
	Glove_vectors = KeyedVectors.load_word2vec_format('../asgn2data/Glove.6B.300d.txt', binary=False)
	ret1=np.zeros((25000, 300))
	# ret2=np.zeros((25000, 300))
	
	tfidf1, tfidf2 = getTFIDF()

	for i in range(Xtr.shape[0]):
		k=0
		temp = Xtr.getrow(i)
		temp = temp.toarray()
		t = tfidf1.getrow(i)
		t = t.toarray()
		for j in range(Xtr.shape[1]):
			if(temp[0][j]):
				if(vocab[j] in Glove_vectors.vocab):
					ret1[i] += t[0][j]*Glove_vectors[vocab[j]]
					k+=t[0][j]
		ret1[i] = ret1[i]/k
		print "  Iteration %d out of %d\r" % (i,Xtr.shape[0]) ,

	print("")
	print('Train features prepared')

	# for i in range(Xts.shape[0]):
	# 	k=0
	# 	temp = Xts.getrow(i)
	# 	temp = temp.toarray()
	# 	t = tfidf2.getrow(i)
	# 	t = t.toarray()
	# 	for j in range(Xts.shape[1]):
	# 		if(temp[0][j]):
	# 			if(vocab[j] in Glove_vectors.vocab):
	# 				ret2[i] += t[0][j]*Glove_vectors[vocab[j]]
	# 				k+=t[0][j]
	# 	ret2[i] = ret2[i]/k
	# 	print "  Iteration %d out of %d\r" % (i,Xts.shape[0]) ,

	np.save('train_weighted_glove_shuffled.npy', ret1)
	# np.save('test_weighted_glove.npy', ret2)
	# print("")


	# ret1 = np.load('train_weighted_glove.npy')
	ret2 = np.load('test_weighted_glove.npy')

	return ret1, ret2









##### CLASSIFIERS ######


# Bernouilli Naive Bayes
def bernoulliNaiveBayes(x, xt):
	clf = BernoulliNB()
	y = 2*(Ytr>5)-1
	yt = 2*(Yts>5)-1
	# print(y)
	clf.fit(x, y)
	yp = clf.predict(xt)
	print((sum(yp==yt)*1.0)/len(yp))

# Multinouilli Naive Bayes
def multinoulliNaiveBayes(x, xt):
	clf = MultinomialNB()
	y = 2*(Ytr>5)-1
	yt = 2*(Yts>5)-1
	clf.fit(x, y)
	yp = clf.predict(xt)
	print((sum(yp==yt)*1.0)/len(yp))

# Logistic Regression
def logisticRegression(x, xt):
	clf = LogisticRegression()
	y = (Ytr>5)
	yt = (Yts>5)
	clf.fit(x, y)
	yp = clf.predict(xt)
	print((sum(yp==yt)*1.0)/len(yp))

# Support Vector Machine
def supportVectorMachine(x, xt):
	clf = svm.SVC()
	y = 2*(Ytr>5)-1
	yt = 2*(Yts>5)-1
	clf.fit(x, y)
	yp = clf.predict(xt)
	print((sum(yp==yt)*1.0)/len(yp))


# FeedForward Neural Network
def feedForwardNeuralNetwork(x, xt):
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,10), random_state=1)
	y = 2*(Ytr>5)-1
	yt = 2*(Yts>5)-1
	clf.fit(x, y)
	yp = clf.predict(xt)
	print((sum(yp==yt)*1.0)/len(yp))


def rnnLSTM(x, xt):
	y = (Ytr>5)
	yt = (Yts>5)

	x = x.reshape(x.shape[0],1,x.shape[1])
	xt = xt.reshape(xt.shape[0],1,xt.shape[1])
	# print(x.shape[0], x.shape[1])
	model = Sequential()
	model.add(LSTM(3, dropout=0.2, recurrent_dropout=0.2, input_shape=(1,x.shape[2])))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	print('Train...')
	batch_size = 100
	model.fit(x, y, batch_size=batch_size, epochs=1, validation_data=(xt, yt))
	score, acc = model.evaluate(xt, yt, batch_size=batch_size)
	print('Test accuracy:', acc)



##### MAIN ######

def main():


	x, xt=getGloveWeightedAvg()
	rnnLSTM(x, xt)
	x, xt = getGloveAvg()
	rnnLSTM(x,xt)
	x, xt = getWord2VecAvg()
	rnnLSTM(x,xt)
	x, xt = getWord2VecweightedAvg()
	rnnLSTM(x,xt)


if __name__ == '__main__':
	main()
