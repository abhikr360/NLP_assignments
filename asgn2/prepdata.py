import numpy as np
from sklearn.datasets import load_svmlight_file

traindatafile = "../asgn2data/aclImdb/train/labeledBow.feat"
testdatafile = "../asgn2data/aclImdb/test/labeledBow.feat"


tr_data = load_svmlight_file(traindatafile)
Xtr = tr_data[0].toarray();
Ytr = tr_data[1];

print(Xtr.shape)
print(Xtr[0])

print(Ytr.shape)
print(Ytr[0])
