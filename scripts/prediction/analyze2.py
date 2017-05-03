
import os
import sys
import traceback
import collections
import numpy as np


#metric_names = ["P-Fm", "P-Prec", "P-Recall", "Fm", "Prec", "Recall", "DRD", "PSNR", "Accuracy"]
metric_names = ["Fm", "Prec", "Recall", "DRD", "PSNR", "Accuracy"]
#metric_names = ["P-Fm", "P-Prec", "P-Recall", "Fm", "Prec", "Recall", "PSNR", "Accuracy"]
display_metric_names = ["P-Fm", "Fm", "DRD", "PSNR"]
robust = False
_max = False
remove_outliers = False
print_diff = False
diffs_as_std = True

def avg(l):
	if remove_outliers and len(l) > 2:
		return (sum(l) - max(l) - min(l)) / float(len(l) -2) if len(l) else -999 
	else:
		return sum(l) / float(len(l)) if len(l) else -999 

if robust:
	def avg(l):
		return float(np.median(l))

if _max:
	def avg(l):
		return float(np.max(l))


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
		metrics = {'val': collections.defaultdict(list),
				   'test':       collections.defaultdict(list),
				   'train':      collections.defaultdict(list)}
		at_least_one = False
		for ssdir in os.listdir(rdir):
			rrdir = os.path.join(rdir, ssdir)
			for split in ['train', 'val', 'test']:
				result_file = os.path.join(rrdir, '%s_summary.txt' % split)
				if not os.path.exists(result_file):
					print "%s does not exist" % result_file
					continue

				tokens = open(result_file).read().split()
				try:
					del tokens[0]
					tokens = map(float, tokens)
					if remove_outliers and tokens[0] < 80:
						continue
					for idx, metric in enumerate(metric_names):
						metrics[split][metric].append(tokens[idx])

					at_least_one = True
				except Exception as e:
					print "Error parsing %s" % result_file
					#raise
					continue
		if not at_least_one:
			continue

		_l = [sdir + " (%d)" % len(metrics['val']['P-Fm'])]
		_diffs = [sdir + "_diff"]
		labels = ['Experiments:']
		try:
			for metric in display_metric_names:
				for split, split_label in zip(['val', 'train', 'test'], ['V', 'T', 'Z']):
				#for split, split_label in zip(['val', 'train'], ['V', 'T']):
					_l.append(avg(metrics[split][metric]))
					if diffs_as_std:
						_diffs.append(np.std(metrics[split][metric]) if metrics[split][metric] else -999)
					else:
						_diffs.append(max(metrics[split][metric]) - min(metrics[split][metric]) if 
							metrics[split][metric] else -999)
					labels.append((split_label + '_' + metric)[:6])
		except:
			print metric, split
			raise

		l.append( tuple(_l))
		diffs.append(tuple(_diffs))
except:
	print sdir, ssdir
	raise

l, diffs = zip(*sorted(zip(l, diffs), key=lambda tup: tup[0][1], reverse=True))  # sort on Val P-Fm

max_display_len = max(len("Experiment"), max(map(lambda x: len(x[0]), diffs if print_diff else l)))

format_str = "%-" + str(max_display_len) + "s  " + " ".join(["%7.2f"] * (len(l[0]) - 1))

format_str_labels = "%-" + str(max_display_len) + "s    " + "%-6s  " * (len(labels) - 1)
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
		
			

