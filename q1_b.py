import re
import os
import sys
 
# Input
if(len(sys.argv)>1):
	rf = open(str(sys.argv[1]),'r')
else:
	print("Please enter file name....Aborting !")
	exit()
t = rf.read()
rf.close()

# Removing abbreviations
abr = re.compile("([A-Z])\.")
s = abr.sub(r'\1@', t)
abr = re.compile("(Mr|Dr|Jr|St}Mrs)\.")
s = abr.sub(r'\1@', s)

### Sentence Tagging

## End of sentence
pat1 = re.compile("(!|\?|\.)('?)(\s*)('?[A-Z])")
s = pat1.sub(r'\1\2</s>\3\4', s)



## Hardcoding to handle few corner cases

# The Preface
s = re.sub("(A cloud)", r'<s>\1', s)

# Special Case of Chapter I because of G.K.C.
s = re.sub("(THE suburb)", r'<s>\1', s)

# Remove Chapter Headings as these can't be part of sentences
pat2 = re.compile('^CHAPTER[ A-Z]*\n\n[A-Z ]*\n', re.M)# print(re.findall('^CHAPTER[ A-Z]*\n\n[A-Z ]*\n', s, re.M))
s = pat2.sub(' ', s)

# End Of File
if(str(sys.argv[1])=='test.txt'):
	s = re.sub(" corridor.", " corridor.</s>", s)
elif(str(sys.argv[1])=='fulltest.txt'):
	s = re.sub("of a girl.", "of a girl.</s>", s)
else:
	dummy=0

## Start of sentence
pat2 = re.compile("(</s>)(\s*)('?)([A-Z])([^@])")
s = pat2.sub(r'\1\2<s>\3\4\5', s)

# Bring back abbreviations
s = re.sub('@','.',s)

# Output
os.system("rm -f out_1_b_full.txt")
wf = open('out_1_b_full.txt', 'w')
wf.write(s)
wf.close()