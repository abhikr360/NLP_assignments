import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.naive_bayes import GaussianNB
from gensim.models.keyedvectors import KeyedVectors



traindatafile = "../asgn2data/aclImdb/train/labeledBow.feat"
testdatafile = "../asgn2data/aclImdb/test/labeledBow.feat"
tr_data = load_svmlight_file(traindatafile)
Xtr = tr_data[0].toarray();
Ytr = tr_data[1];

################# DOC representations

# Binary bag of words

def getBinaryBagOfWords(Xtr):
	Xtr[Xtr>0] = 1
	# print(np.array_equal(Xtr[0],Xtr[1]))
	return Xtr


# Normalized Term frequency

def getNormalizedTFReprsentation(Xtr):
	temp = np.sum(Xtr, axis=1)
	Xtr = ((((Xtr*1.0).T))/temp).T
	return Xtr

# TFIDF

def getTFIDF(Xtr, D=25000):
	temp = getBinaryBagOfWords(Xtr)
	temp = np.sum(temp, axis=0)
	temp = (D*1.0)/temp
	temp=np.log(temp)
	Xtr = getNormalizedTFReprsentation(Xtr)
	Xtr = Xtr*temp
	return Xtr

# Word2Vec Avg 

def getWord2VecAvg(Xtr, D=25000):
	vocab = [line.rstrip('\n') for line in open('../asgn2data/aclImdb/imdb.vocab')]
	Word2Vec_word_vectors = KeyedVectors.load_word2vec_format('../asgn2data/word2vec.bin', binary=True)
	ret=np.zeros((D, 300))
	k=0
	for i in range(Xtr.shape[0]):
		for j in range(Xtr[i].shape[0]):
			if(Xtr[i][j]):
				if(vocab[j] in Word2Vec_word_vectors.vocab):
					ret[i]+= Word2Vec_word_vectors[vocab[j]]
					k+=1

	return ret/k


def getGloveAvg(Xtr, D=25000):
	vocab = [line.rstrip('\n') for line in open('../asgn2data/aclImdb/imdb.vocab')]
	Word2Vec_word_vectors = KeyedVectors.load_word2vec_format('../asgn2data/Glove.6B.300d.txt', binary=False)
	ret=np.zeros((D, 300))
	k=0
	for i in range(Xtr.shape[0]):
		for j in range(Xtr[i].shape[0]):
			if(Xtr[i][j]):
				if(vocab[j] in Word2Vec_word_vectors.vocab):
					ret[i]+= Word2Vec_word_vectors[vocab[j]]
					k+=1

	return ret/k







######### Classifiers
def bernoulliNaiveBayes(X, Y):
	clf = BernoulliNB()
	clf.fit(X, Y)
	return clf

def main():
	x=getWord2VecAvg(Xtr)
	print(x[0])
	print(x[3])
	print(x[0].shape)
	print(x.shape)



if __name__ == '__main__':
	main()