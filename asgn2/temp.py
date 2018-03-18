from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec


s1 = TaggedDocument(words=['oh', 'do', 'your', 'research'],tags=['Sent_1'])
s2 = TaggedDocument(words=['I', 'am', 'not', 'hero'],tags=['Sent_2'])
s3 = TaggedDocument(words=['I', 'am', 'high', 'functioning','sociopath'],tags=['Sent_3'])

files = ['train-pos.txt', 'train-neg.txt', 'test-pos.txt', 'test-neg.txt']

documents=[]
total_words=0
for f in file:
	path = "../asgn2data/aclImdb/" + f
	count=0
	with open(f, 'r') as curfile:
		lines = curfile.read().splitlines()
	for line in lines:
		words = nltk.word_tokenize(line)
		tags = [f + str(count)]
		count += 1
		documents.append(TaggedDocument(words=words,tags=tags))


# documents = [s1, s2, s3]
model = Doc2Vec(documents, vector_size=300, window=5, min_count=1, workers=4)
# model.build_vocab(documents)
model.train(documents, total_examples=len(documents), epochs=2)
# print(model['Sent_2'])
model.save('my_model.doc2vec')
# load the model back
model_loaded = Doc2Vec.load('my_model.doc2vec')