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

# ADD dummy in beginning and end
t = '\n'+t + '\n'

# To find highlighted words
pat1 = re.compile("(\s+)'(\w+)'([\s,;:\.!-]+)")
s = pat1.sub(r'\1@\2@\3', t)

# Removing apostrophe
apo = re.compile("(\w)'(\w)")
s = apo.sub(r'\1@\2', s)

# To get all double quotes
s = re.sub("'",'"', s)

# To get back single quotes
s = re.sub('@', "'", s)

# Remove dummy newlines
s = s[1:]
s = s[:-1]

# Output
os.system("rm -f out_1_a.txt")
wf = open('out_1_a.txt', 'w')
wf.write(s)
wf.close() 