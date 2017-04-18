
import sys
import math

l = list()
iter_num = -1
for line in open(sys.argv[1]).readlines():
	tokens = line.split()
	if iter_num >= 0:
		loss = float(tokens[-1])
		if not math.isnan(loss):
			l.append( (loss, iter_num) )
	if "Testing net" in line:
		for idx, token in enumerate(tokens):
			if token == 'Iteration':
				iter_num = int(tokens[idx+1][:-1])  # remove comma
	else:
		iter_num = -1
			

l.sort()
if l:
	print l[0][1]  # iter_num of lowest loss
