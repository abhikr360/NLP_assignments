import re
import numpy as np
import os
import scipy as sp
from sklearn.neural_network import MLPClassifier
import sys

## Input

# Train file
if(len(sys.argv)>1):
	rf = open(str(sys.argv[1]),'r')
else:
	rf = open('out_1_b.txt', 'r')

train = rf.read()
rf.close()	

# Test file
if(len(sys.argv)>2):
	rf = open(str(sys.argv[2]),'r')
else:
	rf = open('out_1_b_full.txt', 'r')

test = rf.read()
rf.close()


# We care about only end of sentence in this problem
train = re.sub("<s>", "", train)
test = re.sub("<s>", "", test)

# test = "the game is on.</s> I gave C. J. Mary money.</s> He said, 'I dunno anything man.'</s> Let him go.</s> Let her go.</s> The game is never over J. Watson.</s> 'Are there new players?', he asked.</s> He counter asked, 'Do u understand this?'</s> The reply was in negation.</s>"

# Hyperparameters
window=2 #2
D=24
mid=D/2


# Global constants
punctuation = ['.', '?', '!']
dic = {'la' : 0,'ua' : 1, '.' : 2, '?' : 3, '!' : 4, ',' : 5, '-' : 6, "'" : 7, ';' : 8, ':' : 9, ' ' : 10, '\n' : 11}




############################          MAP
# small alphabet       0
# capital alphabet     1
# period               2
# question mark        3
# exclamation mark     4
# comma                5
# hyphen               6
# single quote         7
# semi-colon           8
# colon                9
# space				   10
# newline              11
###########################

# We will use Non-Weighted Multi-Hot representation


# A function to prepare feature vectors from labelled data
def prep_feature(t):
	n=len(t)
	X = np.zeros(D)
	y = np.array([0])

	for m in re.finditer('\.|\?|!', t):
		i = m.start()
		flag=0
		sq=0
		if( i+1<n and  t[i+1]=='<'):
			flag=1
			sq=0
		elif (i+1<n and t[i+1]=="'" and i+2<n and t[i+2] == '<'):
			flag=1
			sq=1
		else:
			flag=0
			sq=0

		j=1
		f = np.full(D, 0)

		while j<=window and i-j>0:
			if t[i-j].islower():
				f[dic['la']] = 1
			elif t[i-j].isupper():
				f[dic['ua']] = 1
			elif t[i-j] in dic:
				f[dic[t[i-j]]] = 1
			elif t[i-j] == '>' :
				break
			else:
				dummy=0
			j=j+1
		
		if(i+1<n):
			if sq:
				f[mid+dic[t[i+1]]]=1
				j=2
				i=i+4
			else:
				i=i+4
				j=1

			while j<=window and i+j<n:
				if t[i+j].islower():
					f[mid+dic['la']] = 1
				elif t[i+j].isupper():
					f[mid+dic['ua']] = 1
				elif t[i+j] in dic:
					f[mid+dic[t[i+j]]] = 1
				elif t[i+j] == '<' :
					break
				else:
					dummy=0
				j=j+1
		y = np.append(y, [flag])
		X = np.vstack((X, f))



	X=X[1:]
	y=y[1:]

	return X,y


# Getting feature vector
Xtrain, Ytrain = prep_feature(train)
Xtest, Ytest = prep_feature(test)

# Training the model
clf = MLPClassifier(solver='lbfgs', alpha=1e-6,hidden_layer_sizes=(12,4), random_state=1)
clf.fit(Xtrain, Ytrain)                                            #12, 4

# Predicting using trained model
Ypred = clf.predict(Xtest)
Ypred[Ypred.shape[0]-1] = 1


print("Indices of difference between Ytest and Ypred:"),
a=np.where(np.logical_not(np.equal(Ypred, Ytest)))
print(a)

print("Accuracy : "),
print((np.sum(Ypred == Ytest)*1.0)/Ytest.shape)




