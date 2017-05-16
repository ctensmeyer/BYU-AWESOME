
import os
import sys
import numpy as np
import cv2


def h_mean(x1, x2):
	return 2 * x1 * x2 / (x1 + x2) if (x1 + x2) else 0

def invert(im):
	return np.max(im) - im

def measure_acc(im, gt):
	num_errors = np.count_nonzero(im - gt)	
	total = im.shape[0] * im.shape[1] 
	num_correct = total - num_errors
	return 100 * float(num_correct) / total


def measure_psnr(im, gt):
	diff = im - gt
	mse = np.mean(diff * diff)
	_max = np.max(gt)
	return 10 * np.log10(_max * _max / mse)


def measure_drd(im, gt):
	im = im / max(im.max(), 1)
	gt = gt / max(gt.max(), 1) 
	diff = (im - gt)
	gt = gt 
	im = im 

	# compute weight matrix
	W = np.zeros((5,5))
	for i in [-2, -1, 0, 1, 2]:
		for j in [-2, -1, 0, 1, 2]:
			if i or j:
				W[i+2,j+2] = 1 / np.sqrt(i*i + j*j)
	s = W.sum()
	W = W / s

	total = 0
	locs = np.where(diff != 0)
	S = locs[0].shape[0]
	for k in xrange(S):
		h,w = locs[0][k], locs[1][k]
		dist = 0
		for i in [-2, -1, 0, 1, 2]:
			for j in [-2, -1, 0, 1, 2]:
				if h + i >= 0 and h + i < gt.shape[0] and w + j >= 0 and w + j < gt.shape[1]:
					dist += W[i+2,j+2] * (1 if (im[h,w] - gt[h+i,w+j]) else 0)
		total += dist
			
	numer = total

	num_non_uniform = 0
	blocks = 0
	h = 0
	while h + 8 <= gt.shape[0]:
		w = 0
		while w + 8 <= gt.shape[1]:
			sub_im = gt[h:h+8,w:w+8]
			a = sub_im.mean()
			if a != 0 and a != 255:
				num_non_uniform += 1
			blocks += 1
			w += 8
		h += 8
	return numer / float(num_non_uniform)


def measure_fmeasure(im, gt):
	tp = np.logical_and(im != 0, gt != 0).sum()
	tn = np.logical_and(im == 0, gt == 0).sum()

	fp = np.logical_and(im != 0, gt == 0).sum()
	fn = np.logical_and(im == 0, gt != 0).sum()

	assert tp + tn + fp + fn == im.shape[0] * im.shape[1]
	recall = 100 * float(tp) / (tp + fn) if (tp + fn) else 0
	precision = 100 * float(tp) / (tp + fp) if (tp + fp) else 0
	f_measure = h_mean(recall, precision)
	return f_measure, precision, recall


def get_metrics(predict_fn, gt_fn):
	predict_im = cv2.imread(predict_fn, -1).astype(np.int32)
	try:
		gt_im = cv2.imread(gt_fn, -1).astype(np.int32)
	except:
		print "Error loading %s" % gt_fn
		raise

	assert predict_im.shape == gt_im.shape

	accuracy = measure_acc(predict_im, gt_im)
	#psnr = measure_psnr(predict_im, gt_im)
	#drd = measure_drd(predict_im, gt_im)
	f, p, r = measure_fmeasure(predict_im, gt_im)

	#return f, p, r, drd, psnr, accuracy
	return f, p, r, accuracy

def main(predict_dir, gt_dir, out_file, summary_file):
	fd = open(out_file, 'w')
	all_metrics = list()
	print predict_dir
	for fn in os.listdir(predict_dir):
		print fn
		predict_fn = os.path.join(predict_dir, fn)
		gt_fn = os.path.join(gt_dir, fn)
		base = os.path.splitext(fn)[0]
		metrics = get_metrics(predict_fn, gt_fn)
		all_metrics.append(metrics)
		fd.write("%s  %s\n" % (fn, "  ".join(map(lambda f: "%.4f" % f, metrics))))

	fd.close()

	# summary stuff
	fd = open(summary_file, 'w')
	avg_metrics = np.mean(all_metrics, axis=0)
	fd.write("avg:  %s\n" % "  ".join(map(lambda f: "%.4f" % f, avg_metrics)))
	fd.close()
		

if __name__ == "__main__":
	predict_dir = sys.argv[1]
	gt_dir = sys.argv[2]
	out_file = sys.argv[3]
	summary_file = sys.argv[4]
	main(predict_dir, gt_dir, out_file, summary_file)
	
