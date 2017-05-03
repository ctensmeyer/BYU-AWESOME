
import os
import sys
import traceback
import collections
import numpy as np


robust = False
print_diff = False
diffs_as_std = True

if robust:
	def avg(l):
		return float(np.median(l))
else:
	def avg(l):
		return sum(l) / float(len(l)) if len(l) else 999 


l = []
diffs = []

if "diff" == sys.argv[1]:
	print_diff = True
	_dir = sys.argv[2]
else:
	_dir = sys.argv[1]

try:
	for sdir in os.listdir(_dir):
		rdir = os.path.join(_dir, sdir)
		metrics = {'Validation': collections.defaultdict(list),
				   'Test':       collections.defaultdict(list),
				   'Train':      collections.defaultdict(list)}
		at_least_one = False
		for ssdir in os.listdir(rdir):
			rrdir = os.path.join(rdir, ssdir)
			result_file = os.path.join(rrdir, 'results.txt')
			if not os.path.exists(result_file):
				print "%s does not exist" % result_file
				continue
			at_least_one = True
			lines = open(result_file).readlines()
			idx = 0
			cur_split = None
			try:
				while idx < len(lines):
					line = lines[idx]
					if line.strip() == "":
						pass
					elif line.startswith('Validation'):
						cur_split = 'Validation'
					elif line.startswith('Test'):
						cur_split = 'Test'
					elif line.startswith('Train'):
						cur_split = 'Train'
					else:
						tokens = line.split()
						metrics[cur_split][tokens[0]].append(float(tokens[2]))
					idx += 1
			except Exception as e:
				print cur_split, tokens[0]
				print "Error parsing %s" % result_file
				#raise
				continue
		if not at_least_one:
			continue

		_l = [sdir + " (%d)" % len(metrics['Validation']['num_iters']), 
			avg(metrics['Validation']['num_iters'])]
		if diffs_as_std:
			_diffs = [sdir + "_diff", np.std(metrics['Validation']['num_iters']) if metrics['Validation']['num_iters'] else 999]
		else:
			_diffs = [sdir + "_diff", max(metrics['Validation']['num_iters']) - min(metrics['Validation']['num_iters']) 
				if metrics['Validation']['num_iters'] else 999]
		labels = ['Experiments:', 'Iters']
		try:
			for metric in metrics['Train'].keys():
				for split, split_label in zip(['Validation', 'Test', 'Train'], ['V', 'Z', 'T']):
				#for split, split_label in zip(['Validation', 'Train'], ['V', 'T']):
					_l.append(avg(metrics[split][metric]))
					if diffs_as_std:
						_diffs.append(np.std(metrics[split][metric]) if metrics[split][metric] else 999)
					else:
						_diffs.append(max(metrics[split][metric]) - min(metrics[split][metric]) if 
							metrics[split][metric] else 999)
					labels.append((split_label + '_' + metric)[:8])
		except:
			print metric, split
			raise

		l.append( tuple(_l))
		diffs.append(tuple(_diffs))
except:
	print sdir, ssdir
	raise

#l.sort(key=lambda tup: tup[2])  # sort on Val weighted_fmeasure_loss
l, diffs = zip(*sorted(zip(l, diffs), key=lambda tup: tup[0][2]))  # sort on Val weighted_fmeasure_loss
max_display_len = max(len("Experiment"), max(map(lambda x: len(x[0]), diffs if print_diff else l)))
format_str = "%-" + str(max_display_len) + "s  %6d " + " ".join(["%9.4f"] * (len(l[0]) - 2))

format_str_labels = "%-" + str(max_display_len) + "s   " + "%s  " * (len(labels) - 1)
print format_str_labels % tuple(labels)
for avg, diff in zip(l, diffs):
	try:
		print format_str % avg
		if print_diff:
			print format_str % diff
			print
	except Exception as e:
		print
		print e
		print avg
		print diff
		print
		
			

