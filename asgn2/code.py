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


#### LOAD DATA #####

traindatafile = "../asgn2data/aclImdb/train/labeledBow.feat"
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
	# vocab = [line.rstrip('\n') for line in open('../asgn2data/aclImdb/imdb.vocab')]
	# Word2Vec_word_vectors = KeyedVectors.load_word2vec_format('../asgn2data/word2vec.bin', binary=True)
	# ret1=np.zeros((25000, 300))
	# ret2=np.zeros((25000, 300))
	
	# for i in range(Xtr.shape[0]):
	# 	k=0
	# 	temp = Xtr.getrow(i)
	# 	temp = temp.toarray()
	# 	for j in range(Xtr.shape[1]):
	# 		if(temp[0][j]):
	# 			if(vocab[j] in Word2Vec_word_vectors.vocab):
	# 				ret1[i] += Word2Vec_word_vectors[vocab[j]]
	# 				k+=1
	# 	ret1[i] = ret1[i]/k
	# 	print "  Iteration %d out of %d\r" % (i,Xtr.shape[0]) ,

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

	# np.save('train_simple_word2vec.npy', ret1)
	# np.save('test_simple_word2vec.npy', ret2)

	ret1 = np.load('train_simple_word2vec.npy')
	ret2 = np.load('test_simple_word2vec.npy')
	return ret1, ret2


# # Word2Vec Weighted Avg with tfidf 
# def getWord2VecweightedAvg(Xtr, D=25000):
# 	vocab = [line.rstrip('\n') for line in open('../asgn2data/aclImdb/imdb.vocab')]
# 	Word2Vec_word_vectors = KeyedVectors.load_word2vec_format('../asgn2data/word2vec.bin', binary=True)
# 	ret=np.zeros((D, 300))
	
# 	tfidf = getTFIDF(Xtr)

# 	for i in range(Xtr.shape[0]):
# 		k=0
# 		for j in range(Xtr[i].shape[0]):
# 			if(Xtr[i][j]):
# 				if(vocab[j] in Word2Vec_word_vectors.vocab):
# 					ret[i]+= tfidf[i][j]*Word2Vec_word_vectors[vocab[j]]
# 					k+=1
# 		ret[i]=ret[i]/k

# 	return ret


# # Glove Avg
# def getGloveAvg(Xtr, D=25000):
# 	vocab = [line.rstrip('\n') for line in open('../asgn2data/aclImdb/imdb.vocab')]
# 	Glove_vectors = KeyedVectors.load_word2vec_format('../asgn2data/Glove.6B.300d.txt', binary=False)
# 	ret=np.zeros((D, 300))

# 	for i in range(Xtr.shape[0]):
# 		k=0
# 		for j in range(Xtr[i].shape[0]):
# 			if(Xtr[i][j]):
# 				if(vocab[j] in Glove_vectors.vocab):
# 					ret[i]+= Glove_vectors[vocab[j]]
# 					k+=1
# 		ret[i]=ret[i]/k

# 	return ret

# # Glove Weighted Avg with tfidf 
# def getGloveWeightedAvg(Xtr, D=25000):
# 	vocab = [line.rstrip('\n') for line in open('../asgn2data/aclImdb/imdb.vocab')]
# 	Glove_vectors = KeyedVectors.load_word2vec_format('../asgn2data/Glove.6B.300d.txt', binary=False)
# 	ret=np.zeros((D, 300))
# 	tfidf = getTFIDF(Xtr)

# 	for i in range(Xtr.shape[0]):
# 		k=0
# 		for j in range(Xtr[i].shape[0]):
# 			if(Xtr[i][j]):
# 				if(vocab[j] in Glove_vectors.vocab):
# 					ret[i]+= tfidf[i][j]*Glove_vectors[vocab[j]]
# 					k+=1
# 		ret[i]=ret[i]/k

# 	return ret









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
	y = 2*(Ytr>5)-1
	yt = 2*(Yts>5)-1
	clf.fit(x, y)
	yp = clf.predict(xt)
	print((sum(yp==yt)*1.0)/len(yp))

# # Support Vector Machine
# def supportVectorMachine(X,Y):
# 	clf = svm.SVC()
# 	return clf.fit(X,Y)






##### MAIN ######

def main():

	# Calling Bernouilli Naive Bayes

	x, xt=getWord2VecAvg()
	logisticRegression(x, xt)


if __name__ == '__main__':
	main()
