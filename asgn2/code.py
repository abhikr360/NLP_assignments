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
from gensim.models.doc2vec import Doc2Vec
import glob
import random
import nltk
# from nltk.corpus import stopwords


######   Helper functions   ####
# stop_words = set(stopwords.words('english'))

def sent_normalize_text(text):
    norm_text = text.lower()
    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')
    # Pad punctuation with spaces on both sides
    for char in [ '"', ',', '(', ')', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')
    return norm_text





##### DOCUMENT REPRESENTATIONS #######

# Binary bag of words
def getBinaryBagOfWords(Xtr, Xts):
	print("Using Binary Bag of words")
	xtr = (Xtr !=0)
	xts = (Xts !=0)
	return xtr, xts

# Normalized Term frequency
def getNormalizedTFReprsentation(Xtr, Xts):
	print('Using Normalized Term frequency')
	xtr = normalize(Xtr, norm='l1', axis=1)
	xts = normalize(Xts, norm='l1', axis=1)
	return xtr, xts

# TFIDF
def getTFIDF(Xtr, Xts):
	print("Using TFIDF")
	tfidf_transformer = TfidfTransformer()
	xtr = tfidf_transformer.fit_transform(Xtr)
	xts = tfidf_transformer.fit_transform(Xts)
	return xtr, xts

# Word2Vec Avg 
def getWord2VecAvg(Xtr, Xts):
	print("Using word2vec avg")

	#----------------- Use this if feature vector already prepared -----------------------------

	ret1 = np.load('train_simple_word2vec_shuffled.npy')
	ret2 = np.load('test_simple_word2vec.npy')


	#--------------- Else use this to prepare feature vector ------------------------------------


	# print("loading vocabulary")
	# vocab = [line.rstrip('\n') for line in open('../asgn2data/aclImdb/imdb.vocab')]
	# print("loading word2vec vectors")
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

	# print("")
	# print('Train features prepared')
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
	# print("")

	# print('Test features prepared')
	# np.save('train_simple_word2vec_shuffled.npy', ret1)
	# np.save('test_simple_word2vec.npy', ret2)


	return ret1, ret2

# Word2Vec Weighted Avg with tfidf 
def getWord2VecweightedAvg(Xtr, Xts):

	print("Using word2vec weighted avg with tfidfs")
	#----------------- Use this if feature vector already prepared -----------------------------

	ret1 = np.load('train_weighted_word2vec_shuffled.npy')
	ret2 = np.load('test_weighted_word2vec.npy')



	#--------------- Else use this to prepare feature vector -----------------------------------

	# vocab = [line.rstrip('\n') for line in open('../asgn2data/aclImdb/imdb.vocab')]
	# Word2Vec_word_vectors = KeyedVectors.load_word2vec_format('../asgn2data/word2vec.bin', binary=True)
	# ret1=np.zeros((25000, 300))
	# ret2=np.zeros((25000, 300))

	# tfidf1, tfidf2 = getTFIDF()
	
	# for i in range(Xtr.shape[0]):
	# 	k=0
	# 	temp = Xtr.getrow(i)
	# 	temp = temp.toarray()
	# 	t = tfidf1.getrow(i)
	# 	t = t.toarray()
	# 	for j in range(Xtr.shape[1]):
	# 		if(temp[0][j]):
	# 			if(vocab[j] in Word2Vec_word_vectors.vocab):
	# 				ret1[i] += t[0][j]*Word2Vec_word_vectors[vocab[j]]
	# 				k+=t[0][j]
	# 	ret1[i] = ret1[i]/k
	# 	print "  Iteration %d out of %d\r" % (i,Xtr.shape[0]) ,

	# print("")
	# print('Train features prepared')
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

	# np.save('train_weighted_word2vec_shuffled.npy', ret1)
	# np.save('test_weighted_word2vec.npy', ret2)

	
	return ret1, ret2

# Glove Avg
def getGloveAvg(Xtr, Xts):

	print("Using Glove Averaging")

	#----------------- Use this if feature vector already prepared -----------------------------
	ret1 = np.load('train_simple_glove_shuffled.npy')
	ret2 = np.load('test_simple_glove.npy')


	#--------------- Else use this to prepare feature vector -----------------------------------

	# vocab = [line.rstrip('\n') for line in open('../asgn2data/aclImdb/imdb.vocab')]
	# Glove_vectors = KeyedVectors.load_word2vec_format('../asgn2data/Glove.6B.300d.txt', binary=False)
	# ret1=np.zeros((25000, 300))
	# # ret2=np.zeros((25000, 300))
	
	# for i in range(Xtr.shape[0]):
	# 	k=0
	# 	temp = Xtr.getrow(i)
	# 	temp = temp.toarray()
	# 	for j in range(Xtr.shape[1]):
	# 		if(temp[0][j]):
	# 			if(vocab[j] in Glove_vectors.vocab):
	# 				ret1[i] += Glove_vectors[vocab[j]]
	# 				k+=1
	# 	ret1[i] = ret1[i]/k
	# 	print "  Iteration %d out of %d\r" % (i,Xtr.shape[0]) ,

	# print("")
	# print('Train features prepared')

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

	# np.save('train_simple_glove_shuffled.npy', ret1)
	# np.save('test_simple_glove.npy', ret2)

	

	return ret1, ret2

# Glove Weighted Avg with tfidf 
def getGloveWeightedAvg(Xtr, Xts):

	print("Using Glove Averaging weighted tfidf ")

	#----------------- Use this if feature vector already prepared -----------------------------

	ret1 = np.load('train_weighted_glove_shuffled.npy')
	ret2 = np.load('test_weighted_glove.npy')


	#--------------- Else use this to prepare feature vector -----------------------------------

	# vocab = [line.rstrip('\n') for line in open('../asgn2data/aclImdb/imdb.vocab')]
	# Glove_vectors = KeyedVectors.load_word2vec_format('../asgn2data/Glove.6B.300d.txt', binary=False)
	# ret1=np.zeros((25000, 300))
	# ret2=np.zeros((25000, 300))
	
	# tfidf1, tfidf2 = getTFIDF()

	# for i in range(Xtr.shape[0]):
	# 	k=0
	# 	temp = Xtr.getrow(i)
	# 	temp = temp.toarray()
	# 	t = tfidf1.getrow(i)
	# 	t = t.toarray()
	# 	for j in range(Xtr.shape[1]):
	# 		if(temp[0][j]):
	# 			if(vocab[j] in Glove_vectors.vocab):
	# 				ret1[i] += t[0][j]*Glove_vectors[vocab[j]]
	# 				k+=t[0][j]
	# 	ret1[i] = ret1[i]/k
	# 	print "  Iteration %d out of %d\r" % (i,Xtr.shape[0]) ,

	# print("")
	# print('Train features prepared')

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

	# np.save('train_weighted_glove_shuffled.npy', ret1)
	# np.save('test_weighted_glove.npy', ret2)
	# print("")

	return ret1, ret2

# DOC2VEC
def getDoc2Vec():
	print("Using Doc2Vec")
	model= Doc2Vec.load('my_model.doc2vec')
	postrfiles = glob.glob("../asgn2data/aclImdb/train/pos/*.txt")
	negtrfiles = glob.glob("../asgn2data/aclImdb/train/neg/*.txt")
	postsfiles = glob.glob("../asgn2data/aclImdb/test/pos/*.txt")
	negtsfiles = glob.glob("../asgn2data/aclImdb/test/neg/*.txt")

	x = np.zeros((25000,200))
	xt = np.zeros((25000,200))
	y = np.zeros(25000)
	yt = np.zeros(25000)

	i=0
	for f in postrfiles:
		x[i]=model[f]
		y[i]=10
		i+=1
	for f in negtrfiles:
		x[i]=model[f]
		y[i]=0
		i+=1
	i=0
	for f in postsfiles:
		xt[i]=model[f]
		yt[i]=10
		i+=1
	for f in negtsfiles:
		xt[i]=model[f]
		yt[i]=0
		i+=1

	combined = list(zip(x,y))
	random.shuffle(combined)
	x[:], y[:] = zip(*combined)


	return x, xt, y, yt

# Average of sentence vectors
def sen2VecAvg(algo=5):
	print("Using Avg of sentence vectors")
	model= Doc2Vec.load('my_model_sens.doc2vec')
	postrfiles = glob.glob("../asgn2data/aclImdb/train/pos/*.txt")
	negtrfiles = glob.glob("../asgn2data/aclImdb/train/neg/*.txt")
	postsfiles = glob.glob("../asgn2data/aclImdb/test/pos/*.txt")
	negtsfiles = glob.glob("../asgn2data/aclImdb/test/neg/*.txt")

	x = np.zeros((25000,100))
	xt = np.zeros((25000,100))
	y = np.zeros(25000)
	yt = np.zeros(25000)

	i=0
	for f in postrfiles:
		with open(f, 'r') as curfile:
			data = curfile.read().decode("utf-8")
			data = sent_normalize_text(data)
			sens = nltk.sent_tokenize(data)
			for j in range(len(sens)):
				x[i] += model[f+'SENT_{}'.format(j)]
			x[i]=x[i]/len(sens)
			y[i]=10
			i+=1

	for f in negtrfiles:
		with open(f, 'r') as curfile:
			data = curfile.read().decode("utf-8")
			data = sent_normalize_text(data)
			sens = nltk.sent_tokenize(data)
			for j in range(len(sens)):
				x[i] += model[f+'SENT_{}'.format(j)]
			x[i]=x[i]/len(sens)
			y[i]=0
			i+=1

	i=0
	for f in postsfiles:
		with open(f, 'r') as curfile:
			data = curfile.read().decode("utf-8")
			data = sent_normalize_text(data)
			sens = nltk.sent_tokenize(data)
			for j in range(len(sens)):
				xt[i] += model[f+'SENT_{}'.format(j)]
			xt[i]=xt[i]/len(sens)
			yt[i]=10
			i+=1

	for f in negtsfiles:
		with open(f, 'r') as curfile:
			data = curfile.read().decode("utf-8")
			data = sent_normalize_text(data)
			sens = nltk.sent_tokenize(data)
			for j in range(len(sens)):
				xt[i] += model[f+'SENT_{}'.format(j)]
			xt[i]=xt[i]/len(sens)
			yt[i]=0
			i+=1


	combined = list(zip(x,y))
	random.shuffle(combined)
	x[:], y[:] = zip(*combined)


	return x, xt, y, yt

	




##### CLASSIFIERS ######

# Bernouilli Naive Bayes
def bernoulliNaiveBayes(x, xt, Ytr, Yts):
	print("Using bernoulliNaiveBayes : ")
	clf = BernoulliNB()
	y = 2*(Ytr>5)-1
	yt = 2*(Yts>5)-1
	# print(y)
	clf.fit(x, y)
	yp = clf.predict(xt)
	print("accuracy : ",(sum(yp==yt)*1.0)/len(yp))

# Multinouilli Naive Bayes
def multinoulliNaiveBayes(x, xt, Ytr, Yts):
	print("Using multinoulliNaiveBayes")
	clf = MultinomialNB()
	y = 2*(Ytr>5)-1
	yt = 2*(Yts>5)-1
	clf.fit(x, y)
	yp = clf.predict(xt)
	print("accuracy : ", (sum(yp==yt)*1.0)/len(yp))

# Logistic Regression
def logisticRegression(x, xt, Ytr, Yts):
	print('Using Logistic Regression')
	clf = LogisticRegression()
	y = (Ytr>5)
	yt = (Yts>5)
	clf.fit(x, y)
	yp = clf.predict(xt)
	print('accuracy',(sum(yp==yt)*1.0)/len(yp))

# Support Vector Machine
def supportVectorMachine(x, xt, Ytr, Yts):
	print("Using SVM")
	clf = svm.LinearSVC()
	y = 2*(Ytr>5)-1
	yt = 2*(Yts>5)-1
	clf.fit(x, y)
	yp = clf.predict(xt)
	print('accuracy', (sum(yp==yt)*1.0)/len(yp))

# FeedForward Neural Network
def feedForwardNeuralNetwork(x, xt, Ytr, Yts):
	print("Using feedForwardNeuralNetwork")
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,10), random_state=1)
	y = 2*(Ytr>5)-1
	yt = 2*(Yts>5)-1
	clf.fit(x, y)
	yp = clf.predict(xt)
	print('accuracy', (sum(yp==yt)*1.0)/len(yp))

# RNN with LSTM
def rnnLSTM(x, xt, Ytr, Yts, key=1):
	print("Using RNN with LSTM")
	y = (Ytr>5)
	yt = (Yts>5)

	x = x.reshape(x.shape[0],1,x.shape[1])
	xt = xt.reshape(xt.shape[0],1,xt.shape[1])
	model = Sequential()
	if(key==2):# DOC2VEC
		model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, input_shape=(1,x.shape[2])))
	elif(key==3): # SEN2VEC
		model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(1,x.shape[2])))
	else:# Others
		model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(1,x.shape[2])))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	print('Train...')
	batch_size = 50
	model.fit(x, y, batch_size=batch_size, epochs=5, validation_data=(xt, yt))
	score, acc = model.evaluate(xt, yt, batch_size=batch_size)
	print('Test accuracy:', acc)



##### MAIN ######

def main():

	#### LOAD DATA #####
	print('Loading train data')
	traindatafile = "../asgn2data/aclImdb/train/labeledBow_shuffled.feat"
	tr_data = load_svmlight_file(traindatafile)
	Xtr = tr_data[0];
	Ytr = tr_data[1];

	print('Loading test data')
	testdatafile = "../asgn2data/aclImdb/test/labeledBow.feat"
	ts_data = load_svmlight_file(testdatafile)
	Xts = ts_data[0];
	temp=csr_matrix((25000, 4))
	Xts = hstack([Xts,temp])
	Xts = Xts.tocsr()
	Yts = ts_data[1];



	## UNCOMMENT THE PAIR YOU WANT TO USE


	#---------------     Binary Bag of Words ----------------------------
	
	# x, xt = getBinaryBagOfWords(Xtr, Xts)
	# bernoulliNaiveBayes(x,xt, Ytr, Yts)

	# x, xt = getBinaryBagOfWords(Xtr, Xts)
	# multinoulliNaiveBayes(x, xt, Ytr, Yts)

	# x, xt = getBinaryBagOfWords(Xtr, Xts)
	# logisticRegression(x, xt, Ytr, Yts)

	# x, xt = getBinaryBagOfWords(Xtr, Xts)
	# supportVectorMachine(x,xt, Ytr, Yts)

	# x, xt = getBinaryBagOfWords(Xtr, Xts)
	# feedForwardNeuralNetwork(x, xt, Ytr, Yts)



	#---------------  Normalized Term Frequency -------------------------


	# x, xt = getNormalizedTFReprsentation(Xtr, Xts)
	# multinoulliNaiveBayes(x, xt, Ytr, Yts)

	# x, xt = getNormalizedTFReprsentation(Xtr, Xts)
	# logisticRegression(x, xt, Ytr, Yts)

	# x, xt = getNormalizedTFReprsentation(Xtr, Xts)
	# supportVectorMachine(x,xt, Ytr, Yts)

	# x, xt = getNormalizedTFReprsentation(Xtr, Xts)
	# feedForwardNeuralNetwork(x, xt, Ytr, Yts)






	#---------------   TFIDF representation -------------------------

	# x, xt = getTFIDF(Xtr, Xts)
	# multinoulliNaiveBayes(x, xt, Ytr, Yts)

	# x, xt = getTFIDF(Xtr, Xts)
	# logisticRegression(x, xt, Ytr, Yts)

	# x, xt = getTFIDF(Xtr, Xts)
	# supportVectorMachine(x,xt, Ytr, Yts)

	# x, xt = getTFIDF(Xtr, Xts)
	# feedForwardNeuralNetwork(x, xt, Ytr, Yts)




	# ----------------- Word2Vec Averaging --------------

	# x, xt = getWord2VecAvg(Xtr, Xts)
	# logisticRegression(x, xt, Ytr, Yts)

	# x, xt = getWord2VecAvg(Xtr, Xts)
	# supportVectorMachine(x, xt, Ytr, Yts)

	# x, xt = getWord2VecAvg(Xtr, Xts)
	# feedForwardNeuralNetwork(x, xt, Ytr, Yts)

	# x, xt = getWord2VecAvg(Xtr, Xts)
	# rnnLSTM(x, xt, Ytr, Yts)


	# ---------------- Word2vec avg with tfidf -----------

	# x, xt = getWord2VecweightedAvg(Xtr, Xts)
	# logisticRegression(x, xt, Ytr, Yts)

	# x, xt = getWord2VecweightedAvg(Xtr, Xts)
	# supportVectorMachine(x, xt, Ytr, Yts)

	# x, xt = getWord2VecweightedAvg(Xtr, Xts)
	# feedForwardNeuralNetwork(x, xt, Ytr, Yts)

	# x, xt = getWord2VecAvg(Xtr, Xts)
	# rnnLSTM(x, xt, Ytr, Yts)

	# ----------------- Glove Averaging ------------------

	# x, xt = getGloveAvg(Xtr, Xts)
	# logisticRegression(x, xt, Ytr, Yts)

	# x, xt = getGloveAvg(Xtr, Xts)
	# supportVectorMachine(x, xt, Ytr, Yts)

	# x, xt = getGloveAvg(Xtr, Xts)
	# feedForwardNeuralNetwork(x, xt, Ytr, Yts)

	# x, xt = getGloveAvg(Xtr, Xts)
	# rnnLSTM(x, xt, Ytr, Yts)


	# ----------------- Glove Averaging with TFIDF weights ------------------

	# x, xt = getGloveWeightedAvg(Xtr, Xts)
	# logisticRegression(x, xt, Ytr, Yts)

	# x, xt = getGloveWeightedAvg(Xtr, Xts)
	# supportVectorMachine(x, xt, Ytr, Yts)

	# x, xt = getGloveWeightedAvg(Xtr, Xts)
	# feedForwardNeuralNetwork(x, xt, Ytr, Yts)

	# x, xt = getGloveWeightedAvg(Xtr, Xts)
	# rnnLSTM(x, xt, Ytr, Yts)



	#---------------------- DOC2VEC-------------------------------------

	# x, xt, y, yt = getDoc2Vec()
	# logisticRegression(x, xt, y, yt)

	# x, xt, y, yt = getDoc2Vec() 
	# supportVectorMachine(x, xt, y, yt)

	# x, xt, y, yt = getDoc2Vec() 
	# feedForwardNeuralNetwork(x, xt, y, yt)

	# x, xt, y, yt = getDoc2Vec() 
	# rnnLSTM(x, xt, y, yt, key=2)


	#-------------------- Average of sentence vectors --------------

	# x, xt, y, yt = sen2VecAvg()
	# logisticRegression(x, xt, y, yt)

	# x, xt, y, yt = sen2VecAvg() 
	# supportVectorMachine(x, xt, y, yt)

	# x, xt, y, yt = sen2VecAvg() 
	# feedForwardNeuralNetwork(x, xt, y, yt)

	x, xt, y, yt = sen2VecAvg() 
	rnnLSTM(x, xt, y, yt, key=3)




if __name__ == '__main__':
	main()
