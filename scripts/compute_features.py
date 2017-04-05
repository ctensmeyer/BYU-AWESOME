
import os
import sys
import cv2
import scipy.ndimage as nd
import scipy.ndimage.morphology
import numpy as np
import traceback

def invert(im):
	return 255 - im


# deprecated
def gt_original_to_processed(im):
	im = im / 255
	im = 1 - im
	return im


# deprecated
def gt_processed_to_original(im):
	im = 1 - im
	im = im * 255
	return im


def shave(im, top, bottom, left, right):
	if bottom == 0:
		bottom = im.shape[0]
	else:
		bottom *= -1
	if right == 0:
		right = im.shape[1]
	else:
		right *= -1
	if im.ndim == 3:
		return im[top:bottom,left:right,:]
	else:
		return im[top:bottom,left:right]


def bilateral(im):
	return cv2.bilateralFilter(im, d=100, sigmaColor=100, sigmaSpace=100)


def mean_transform(im, window_size):
	return cv2.blur(im, (window_size, window_size), borderType=cv2.BORDER_REFLECT_101)


def median_transform(im, window_size):
	return cv2.medianBlur(im, window_size)


def otsu(im):
	if im.ndim == 3:
		im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	thresh, result = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return result


# TOO SLOW
## creates a particular sauvola threshold function using K, width, height
#def create_sauvola(K, R, width, height):
#	return lambda im, i, j: sauvola_threshold(im, i, j, K, R, width, height)
#
#
## compute the local sauvola threshold for im[i,j] for the given parameters
#def sauvola_threshold(im, i, j, K, R, width, height):
#	window = im[max(i-width/2, 0):i+width/2, max(j-height/2, 0):j+height/2]
#	u = np.mean(window)
#	std = np.std(window)
#	#return u * (1 - K * (1 - (std / R)) )
#	return u + K * std
#
#
## generic local threshold algorithm using fthresh to calculate the local threshold
#def local_thresh(im, fthresh):
#	out = np.zeros_like(im)
#	for i in xrange(out.shape[0]):
#		for j in xrange(out.shape[1]):
#			thresh = fthresh(im, i, j)
#			if im[i,j] >= thresh:
#				out[i,j] = 255
#	return out
#
#
#def sauvola(im, K=-0.2, R=128, size=79):
#	if im.ndim == 3:
#		im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#	sauvola_func = create_sauvola(K, R, size, size)
#	return local_thresh(im, sauvola_func)


def std_dev_transform(im, window_size):
	if im.ndim == 3:
		size = (window_size, window_size, 1)
	else:
		size = (window_size, window_size)
	# scale to a wider range
	return 3 * nd.generic_filter(im, nd.standard_deviation, size=size, mode='reflect')


def max_transform(im, window_size):
	if im.ndim == 3:
		size = (window_size, window_size, 1)
	else:
		size = (window_size, window_size)
	return nd.maximum_filter(im, size=size, mode='reflect')


def min_transform(im, window_size):
	if im.ndim == 3:
		size = (window_size, window_size, 1)
	else:
		size = (window_size, window_size)
	return nd.minimum_filter(im, size=size, mode='reflect')


def percentile_10_transform(im, window_size):
	if im.ndim == 3:
		size = (window_size, window_size, 1)
	else:
		size = (window_size, window_size)
	return nd.percentile_filter(im, percentile=10, size=size, mode='reflect')


def percentile_25_transform(im, window_size):
	if im.ndim == 3:
		size = (window_size, window_size, 1)
	else:
		size = (window_size, window_size)
	return nd.percentile_filter(im, percentile=25, size=size, mode='reflect')


def canny(im, low=100, high=200):
	return cv2.Canny(im, low, high, L2gradient=True)


def percentile_gray(im):
	hist, bin_edges =  np.histogram(im, bins=256, range=(0,256), density=True)
	cum_histo = np.cumsum(hist)
	cum_histo *= 255
	cum_histo = cum_histo.astype(np.uint8)
	return cum_histo[im]


def percentile(im):
	if im.ndim == 2:
		return percentile_gray(im)
	else:
		b_perc = percentile_gray(im[:,:,0])
		g_perc = percentile_gray(im[:,:,1])
		r_perc = percentile_gray(im[:,:,2])
		return np.concatenate([b_perc[:,:,np.newaxis], g_perc[:,:,np.newaxis], r_perc[:,:,np.newaxis]], axis=2)

def slice(im, axis):
	return im[:,:,axis]

def relative_darkness(im, window_size, threshold=15):
	return relative_darkness2(im, window_size, threshold)


def relative_darkness2(im, window_size, threshold=15, group=None):
	if im.ndim == 3:
		im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	# find number of pixels at least $threshold less than the center value
	def below_thresh(vals):
		center_val = vals[vals.shape[0]/2]
		lower_thresh = center_val - threshold
		return (vals < lower_thresh).sum()

	# find number of pixels at least $threshold greater than the center value
	def above_thresh(vals):
		center_val = vals[vals.shape[0]/2]
		above_thresh = center_val + threshold
		return (vals > above_thresh).sum()
		
	# apply the above function convolutionally
	lower = nd.generic_filter(im, below_thresh, size=window_size, mode='reflect')
	upper = nd.generic_filter(im, above_thresh, size=window_size, mode='reflect')

	# number of values within $threshold of the center value is the remainder
	# constraint: lower + middle + upper = window_size ** 2
	middle = np.empty_like(lower)
	middle.fill(window_size*window_size)
	middle = middle - (lower + upper)

	# scale to range [0-255]
	lower = lower * (255 / (window_size * window_size))
	middle = middle * (255 / (window_size * window_size))
	upper = upper * (255 / (window_size * window_size))

	if group == 'lower':
		return lower
	if group == 'middle':
		return middle
	if group == 'upper':
		return upper

	return np.concatenate( [lower[:,:,np.newaxis], middle[:,:,np.newaxis], upper[:,:,np.newaxis]], axis=2)


def remove_small_ccs(im, min_area=10, structure=np.ones(shape=(3,3), dtype=int)):
	inverted = invert(im)  # 0 is considered to be background
	labeled, num_ccs = nd.label(inverted, structure=structure)
	all_cc_slices = nd.find_objects(labeled)

	for y, x in all_cc_slices:
		area = (y.stop - y.start) * (x.stop - x.start)
		if area < min_area:
			inverted[y,x] = 0

	return invert(inverted)

def create_dilated_recall_weights():
	in_dir = sys.argv[1]
	recall_dir = os.path.join(in_dir, 'recall_weights')
	out_dir = os.path.join(in_dir, 'dilated_recall_weights')
	if not os.path.isdir(out_dir):
		os.makedirs(out_dir)
	func = lambda im: nd.morphology.grey_dilation(im, size=(3,3))
	convert_dir(func, recall_dir, out_dir)


def modify_recall_weights(im):
	binary = np.copy(im)
	non_zero_idx = binary > 0
	binary[non_zero_idx] = 1
	struct = nd.generate_binary_structure(2,2)  # 3x3 full
	dilated = nd.morphology.binary_dilation(binary, struct)
	diff = dilated - binary
	diff_idx = diff != 0
	im[diff_idx] = 5
	return im
	

# sets the border pixels to have a weight of 5 (out of 128)
def create_modified_recall_weights():
	in_dir = sys.argv[1]
	recall_dir = os.path.join(in_dir, 'recall_weights')
	out_dir = os.path.join(in_dir, 'modified_recall_weights')
	if not os.path.isdir(out_dir):
		os.makedirs(out_dir)
	func = modify_recall_weights
	convert_dir(func, recall_dir, out_dir)

def convert_dat(fname, size):
	flat = np.loadtxt(fname)
	return flat.reshape(size)


def convert_dats(im_file, recall_file, precision_file, recall_dir, precision_dir):
	im = cv2.imread(im_file)
	size = im.shape[:2]

	recall_out = os.path.join(recall_dir, os.path.basename(im_file))
	if not os.path.exists(recall_out):
		recall_im = convert_dat(recall_file, size)
		cv2.imwrite(recall_out, 128 * recall_im)  # scale weights for discritization

	precision_out = os.path.join(precision_dir, os.path.basename(im_file))
	if not os.path.exists(precision_out):
		precision_im = convert_dat(precision_file, size) + 1
		cv2.imwrite(precision_out, 128 * precision_im)  # scale weights for discritization

def convert_dats_main():
	in_dir = sys.argv[1]
	recall_dir = os.path.join(in_dir, 'recall_weights')
	try:
		os.makedirs(recall_dir)
	except:
		pass
	precision_dir = os.path.join(in_dir, 'precision_weights')
	try:
		os.makedirs(precision_dir)
	except:
		pass
	dat_dir = os.path.join(in_dir, 'pr_dats')
	for f in os.listdir(dat_dir):
		if not f.endswith('.png'):
			continue
		im_file = os.path.join(dat_dir, f)
		base = os.path.splitext(f)[0]
		recall_file = os.path.join(dat_dir, base + "_RWeights.dat")
		precision_file = os.path.join(dat_dir, base + "_PWeights.dat")
		if not os.path.exists(recall_file):
			#raise Exception("%s does not exist" % recall_file)
			print "%s does not exist" % recall_file
			continue
		if not os.path.exists(precision_file):
			#raise Exception("%s does not exist" % precision_file)
			print "%s does not exist" % precision_file
			continue
		try:
			convert_dats(im_file, recall_file, precision_file, recall_dir, precision_dir)
		except:
			print im_file
			traceback.print_exc()


def create_uniform_weights():
	root_dir = sys.argv[1]
	uniform_recall_dir = os.path.join(root_dir, 'uniform_recall_weights')
	try:
		os.makedirs(uniform_recall_dir)
	except:
		pass
	uniform_precision_dir = os.path.join(root_dir, 'uniform_precision_weights')
	try:
		os.makedirs(uniform_precision_dir)
	except:
		pass

	in_dir = os.path.join(root_dir, 'original_images')
	convert_dir(lambda im: 128 * np.ones_like(im), in_dir, uniform_recall_dir)
	convert_dir(lambda im: 128 * np.ones_like(im), in_dir, uniform_precision_dir)

def create_dilated_baselines():
	root_dir = sys.argv[1]
	for x in [1,3,5,7]:
		out_dir = os.path.join(root_dir, 'baselines_%d' % x)
		try:
			os.makedirs(out_dir)
		except:
			pass

		in_dir = os.path.join(root_dir, 'baselines')
		structure = np.ones((x,x))
		convert_dir(lambda im: nd.morphology.binary_dilation(im/255, structure=structure).astype(np.uint8), in_dir, out_dir)


def convert_dir(func, in_dir, out_dir, force_overwrite=False):
	for f in os.listdir(in_dir):
		in_path = os.path.join(in_dir, f)
		f_base = os.path.basename(f)
		f_base = os.path.splitext(f_base)[0]
		out_path = os.path.join(out_dir, f_base + ".png")
		if os.path.exists(out_path) and not force_overwrite:
			continue

		im = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
		if im is None:
			raise Exception("Image %s could not be read" % in_path)
		processed = func(im)

		cv2.imwrite(out_path, processed)


def convert_file(func, in_file, out_file, gray=False):
	im = cv2.imread(in_file, cv2.IMREAD_UNCHANGED)
	processed = func(im)
	cv2.imwrite(out_file, processed)
	

def process_features1():
	_dir = sys.argv[1]
	in_dir = os.path.join(_dir, "original_images")
	for transform in ['mean', 'median']:
		print transform
		func = globals()[transform + "_transform"]
		for size in [9, 19, 39, 79]:
			print "  ", size
			size_func = lambda im: func(im, size)
			out_dir = os.path.join(_dir, transform, str(size))
			if not os.path.isdir(out_dir):
				os.makedirs(out_dir)
			convert_dir(size_func, in_dir, out_dir)
	
	
def process_features2():
	_dir = sys.argv[1]
	in_dir = os.path.join(_dir, "original_images")
	for transform in ['min', 'max', 'percentile_10', 'percentile_25']:
		print transform
		func = globals()[transform + "_transform"]
		for size in [3, 5, 7, 9]:
			print "  ", size
			size_func = lambda im: func(im, size)
			out_dir = os.path.join(_dir, transform, str(size))
			if not os.path.isdir(out_dir):
				os.makedirs(out_dir)
			convert_dir(size_func, in_dir, out_dir)


def process_features3():
	_dir = sys.argv[1]
	in_dir = os.path.join(_dir, "original_images")
	for transform in ['std_dev']:
		print transform
		func = globals()[transform + "_transform"]
		for size in [3, 5, 7, 9]:
			print "  ", size
			size_func = lambda im: func(im, size)
			out_dir = os.path.join(_dir, transform, str(size))
			if not os.path.isdir(out_dir):
				os.makedirs(out_dir)
			convert_dir(size_func, in_dir, out_dir)


def process_features4():
	_dir = sys.argv[1]
	in_dir = os.path.join(_dir, "original_images")
	for transform in ['bilateral', 'percentile', 'otsu']:
		print transform
		func = globals()[transform]
		out_dir = os.path.join(_dir, transform)
		if not os.path.isdir(out_dir):
			os.makedirs(out_dir)
		convert_dir(func, in_dir, out_dir)


def process_features5():
	_dir = sys.argv[1]
	in_dir = os.path.join(_dir, "original_images")
	for transform in ['relative_darkness2']:
		print transform
		func = globals()[transform]
		for thresh in [10, 20, 40]:
			print "  ", thresh
			for size in [5, 7, 9]:
				print "    ", size
				for group in ['lower', 'middle', 'upper']:
					print "      ", group
					size_func = lambda im: func(im, size, thresh, group)
					out_dir = os.path.join(_dir, transform, str(size), str(thresh), group)
					if not os.path.isdir(out_dir):
						os.makedirs(out_dir)
					convert_dir(size_func, in_dir, out_dir)

def process_features7():
	_dir = sys.argv[1]
	in_dir = os.path.join(_dir, "original_images")
	for transform in ['slice']:
		print transform
		func = globals()[transform]
		for name, axis in [('b', 0), ('g', 1), ('r', 2)]:
			f = lambda im: func(im, axis)
			out_dir = os.path.join(_dir, transform, name)
			if not os.path.isdir(out_dir):
				os.makedirs(out_dir)
			convert_dir(f, in_dir, out_dir)


def process_features6():
	_dir = sys.argv[1]
	in_dir = os.path.join(_dir, "original_images")
	for transform in ['canny']:
		print transform
		func = globals()[transform]
		for low in [75, 100, 125]:
			print "  ", low
			for high in [150, 175, 200]:
				print "    ", high
				size_func = lambda im: func(im, low, high)
				out_dir = os.path.join(_dir, transform, str(low), str(high))
				if not os.path.isdir(out_dir):
					os.makedirs(out_dir)
				convert_dir(size_func, in_dir, out_dir)


def process_features():
	process_features1()
	process_features2()
	process_features3()
	process_features4()
	process_features5()
	process_features6()


def process_gt():
	_dir = sys.argv[1]
	in_dir = os.path.join(_dir, 'original_gt')
	out_dir = os.path.join(_dir, 'processed_gt')
	if not os.path.isdir(out_dir):
		os.makedirs(out_dir)
	convert_dir(gt_original_to_processed, in_dir, out_dir)


def invert_gt():
	_dir = sys.argv[1]
	in_dir = os.path.join(_dir, 'original_gt')
	out_dir = os.path.join(_dir, 'original_gt')
	if not os.path.isdir(out_dir):
		os.makedirs(out_dir)
	convert_dir(invert, in_dir, out_dir, force_overwrite=True)


def crop_parzival():
	in_dir = sys.argv[1]
	out_dir = sys.argv[2]
	if not os.path.isdir(out_dir):
		os.makedirs(out_dir)
	func = lambda im: shave(im, 200, 200, 75, 120)
	convert_dir(func, in_dir, out_dir)


def crop_saint_gall():
	in_dir = sys.argv[1]
	out_dir = sys.argv[2]
	if not os.path.isdir(out_dir):
		os.makedirs(out_dir)
	func = lambda im: shave(im, 500, 700, 225, 225)
	convert_dir(func, in_dir, out_dir)


def crop_rodrigo():
	in_dir = sys.argv[1]
	out_dir = sys.argv[2]
	if not os.path.isdir(out_dir):
		os.makedirs(out_dir)
	func = lambda im: shave(im, 30, 30, 30, 30)
	convert_dir(func, in_dir, out_dir)


def crop_single():
	top = int(sys.argv[3])
	bottom = int(sys.argv[4])
	left = int(sys.argv[5])
	right = int(sys.argv[6])
	func = lambda im: shave(im, top, bottom, left, right)
	convert_file(func, sys.argv[1], sys.argv[2])


def crop_perc():
	_dir = sys.argv[1]
	perc = int(sys.argv[2]) / 100.
	func = lambda im: im[:, :int(im.shape[1] * perc)]
	convert_dir(func, _dir, _dir, True)


def clean_binary_single():
	func = lambda im: remove_small_ccs(im, int(sys.argv[3]))
	convert_file(func, sys.argv[1], sys.argv[2], gray=True)


def clean_binary_parzival():
	func = lambda im: remove_small_ccs(im, 200)
	convert_dir(func, sys.argv[1], sys.argv[2], gray=True)


def clean_binary_saintgall():
	func = lambda im: remove_small_ccs(im, 400)
	convert_dir(func, sys.argv[1], sys.argv[2], gray=True)


def clean_binary_hbr():
	func = lambda im: remove_small_ccs(im, 80)
	convert_dir(func, sys.argv[1], sys.argv[2], gray=True)


def clean_binary_hdlac():
	func = lambda im: remove_small_ccs(im, 100)
	convert_dir(func, sys.argv[1], sys.argv[2], gray=True)
	

if __name__ == "__main__":
	#crop_saint_gall()
	#crop_parzival()
	#crop_rodrigo()
	#crop_single()
	#crop_perc()
	#process_features()
	#process_features1()
	#process_features2()
	#process_features3()
	#process_features4()
	#process_features5()
	#process_features6()
	#process_features7()
	#process_gt()
	#invert_gt()
	#clean_binary_single()
	#clean_binary_parzival()
	#clean_binary_saintgall()
	#clean_binary_hbr()
	#clean_binary_hdlac()
	#convert_dats_main()
	#create_uniform_weights()
	#create_dilated_recall_weights()
	#create_modified_recall_weights()
	create_dilated_baselines()

	#convert_dir(invert, sys.argv[1], sys.argv[2], True)
	#convert_file(bilateral, '/home/chris/Dropbox/test.jpg', '/home/chris/Dropbox/out.png')
	#for size in [3, 5, 7, 9]:
	#	for thresh in [7, 15, 20, 30, 45]:
	#		convert_file(lambda im: relative_darkness(im, size, threshold=thresh), '/home/chris/Dropbox/test.jpg', '/home/chris/Dropbox/out_%d_%d.png' % (size,thresh))
	


