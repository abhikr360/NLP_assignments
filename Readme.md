### q1_a.py ###

This code takes filename as command line argument. First I add a dummy new line at the beginning and end of file.
This will help in case when first or the last word of text is highlighted using
single quote. Then I replace all the apostrophe and highlighted words
by @. After that all the single quotes are converted to double quote. Thus for
the nested quote case, I convert all of them into double quote. Then I
bring back apostrophe and highlighting single quotes. The ouput is saved
in file called 'out_1_a.txt'

### q1_b.py ###

This code takes filename as command line argument. In this question, I
first replace all the periods present in abbreviations and titles with @(for eg. Mr. -> Mr@).
Then I tagged end of sentences using the heuristic that next alphabet
after punctuation is Capital. This might fail sometimes in case of
proper noun but for given documents this heuristic works pretty
decently. Then I hardcoded to work with a corner case of first sentence.
I also removed CHAPTER titles as it wasn't part of any sentence. Then I
tagged beginning of sentences using end tags(i.e. </s>). Finally I bring
back periods pertaining to abbreviations. The output is saved in file
called 'out_1_b.txt'. The output file 'out_1_b_full.txt' is the output
of this code when 'fulltest.txt' is given as input to this program.


### q2.py ###

This code takes two command line arguments first is the name of train
file and second is the name of test file. Both should be tagged using
program q2.py. In case they are ommitted, by default this program
considers 'out_1_b.txt' as train file and 'out_1_b_full.txt' as test
file.
First I remove "<s>" from file we care about only end of sentence in this
problem. Window size is hyperparameter. I tuned it using cross-validation.
I create 24 dimensional multihot representation for each punctuation
mark. This multi-hot representation depends on the characters present
inside context window(both left and right). There are 12 such characters
possible(Hence 12+12=24). I maintain a dictionary of all these characters and if a character is found in the context of
punctuaution mark, I turn the bit corresponding to 1. Presence of </s>
nearby helps in determining the label(0/1). After preparing training
data, I learn a model using sklearn.neural_network(http://scikit-learn.org/stable/modules/neural_networks_supervised.html)
 and then predict the labels for test data. Finally I report accuracy of learnt labels.
 I also report indices where our learnt model mispredicted.
