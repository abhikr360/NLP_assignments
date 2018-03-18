from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
import glob, os
import nltk
import os
import numpy as np
import glob

# s1 = TaggedDocument(words=['oh', 'do', 'your', 'research'],tags=['Sent_1'])
# s2 = TaggedDocument(words=['I', 'am', 'not', 'hero'],tags=['Sent_2'])
# s3 = TaggedDocument(words=['I', 'am', 'high', 'functioning','sociopath'],tags=['Sent_3'])

# files = ['train-pos.txt', 'train-neg.txt', 'test-pos.txt', 'test-neg.txt']

postrfiles = glob.glob("../asgn2data/aclImdb/train/pos/*.txt")
# print(len(postrfiles))
negtrfiles = glob.glob("../asgn2data/aclImdb/train/neg/*.txt")
# print(len(negtrfiles))
postsfiles = glob.glob("../asgn2data/aclImdb/test/pos/*.txt")
# print(len(postsfiles))
negtsfiles = glob.glob("../asgn2data/aclImdb/test/neg/*.txt")
# print(len(negtsfiles))

# folders = [postrfiles, postsfiles, negtrfiles, negtsfiles]

documents=[]

def normalize_text(text):
    norm_text = text.lower()
    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')
    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')
    return norm_text

# print(len(postrfiles))

i=0
for file in postrfiles:
	os.system("perl -pi -e 's/[^[:ascii:]]//g' {}".format(file))

	with open(file, 'r') as curfile:
		data = curfile.read().decode("utf-8")
		data = normalize_text(data)
		words = nltk.word_tokenize(data)
		tags = [file]
		documents.append(TaggedDocument(words=words,tags=tags))
	print "  Iteration %d\r" % (i) ,
	i+=1
	
np.save('documents.npy', documents)
print(" 1 done")

i=0
for file in postsfiles:
	os.system("perl -pi -e 's/[^[:ascii:]]//g' {}".format(file))
	with open(file, 'r') as curfile:
		data = curfile.read().decode("utf-8")
		data = normalize_text(data)
		words = nltk.word_tokenize(data)
		tags = [file]
		documents.append(TaggedDocument(words=words,tags=tags))
	print "  Iteration %d\r" % (i) ,
	i+=1
	
np.save('documents.npy', documents)
print(" 2 done")

i=0
for file in negtrfiles:
	os.system("perl -pi -e 's/[^[:ascii:]]//g' {}".format(file))
	with open(file, 'r') as curfile:
		data = curfile.read().decode("utf-8")
		data = normalize_text(data)
		words = nltk.word_tokenize(data)
		tags = [file]
		documents.append(TaggedDocument(words=words,tags=tags))
	print "  Iteration %d\r" % (i) ,
	i+=1
np.save('documents.npy', documents)
print("3 done")

i=0
for file in negtsfiles:
	os.system("perl -pi -e 's/[^[:ascii:]]//g' {}".format(file))
	with open(file, 'r') as curfile:
		data = curfile.read().decode("utf-8")
		data = normalize_text(data)
		words = nltk.word_tokenize(data)
		tags = [file]
		documents.append(TaggedDocument(words=words,tags=tags))
	print "  Iteration %d\r" % (i) ,
	i+=1
np.save('documents.npy', documents)
print("4 done")

model = Doc2Vec(documents, vector_size=300, window=5, min_count=1, workers=4)
# model.build_vocab(documents)
model.train(documents, total_examples=len(documents), epochs=10)
# print(model['Sent_2'])
print("Trained")
model.save('my_model.doc2vec')
# load the model back
model_loaded = Doc2Vec.load('my_model.doc2vec')