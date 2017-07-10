#!/usr/bin/python

import os
import sys
import collections
import argparse
import numpy as np
import caffe
import cv2
import random
import scipy.ndimage as nd
from adaptive_gt_raw import adaptive_gt



def safe_mkdir(_dir):
	try:
		os.makedirs(_dir)
	except:
		pass

def xor_image(im1, im2, invert=True):
	out_image = np.zeros(im1.shape + (3,), np.uint8)

	for y in xrange(im1.shape[0]):
	    for x in xrange(im1.shape[1]):
			if im1[y,x]:
				if im2[y,x]:
					# white on white
					if invert:
						out_image[y,x] = (0,0,0)
					else:
						out_image[y,x] = (255,255,255)
				else:
					# white on black
					out_image[y,x] = (255,0,0)
			else:
				if im2[y,x]:
					# black on white
					out_image[y,x] = (0,255,0)
				else:
					# black on black
					if invert:
						out_image[y,x] = (255,255,255)
					else:
						out_image[y,x] = (0,0,0)
	return out_image


def dump_debug(out_dir, data):
	pred_dir = os.path.join(out_dir, 'probs')
	pred_original_size_dir = os.path.join(out_dir, 'probs_original_size')
	binary_dir = os.path.join(out_dir, 'binary')
	gt_dir = os.path.join(out_dir, 'adaptive_gt')
	gt_xor_dir = os.path.join(out_dir, 'gt_xor')
	pred_xor_dir = os.path.join(out_dir, 'pred_xor')

	safe_mkdir(pred_dir)
	safe_mkdir(gt_dir)
	safe_mkdir(gt_xor_dir)
	safe_mkdir(pred_xor_dir)
	safe_mkdir(pred_original_size_dir)
	for idx in xrange(len(data['original_images'])):
		fn = data['filenames'][idx]
		probs = data['probs'][idx] 
		binary = data['binary'][idx] 
		gt = data['current_baselines'][idx] 
		original_gt = data['baselines'][idx]
		original_size = data['original_size'][idx]

		gt_diff = xor_image(gt, original_gt)
		binary_diff = xor_image(binary, gt)

		original_size_probs = cv2.resize(probs, original_size)

		cv2.imwrite(os.path.join(pred_dir, fn), (255 * probs).astype(np.uint8))
		cv2.imwrite(os.path.join(pred_original_size_dir, fn), (255 * original_size_probs).astype(np.uint8))
		cv2.imwrite(os.path.join(binary_dir, fn), 255 * binary)
		cv2.imwrite(os.path.join(gt_dir, fn), (255 * gt).astype(np.uint8))
		cv2.imwrite(os.path.join(gt_xor_dir, fn), gt_diff)
		cv2.imwrite(os.path.join(pred_xor_dir, fn), binary_diff)


def predict(network, ims, output_blob, args):
	idx = 0
	#responses = np.zeros( (len(ims, 1, ims[0].shape[1], ims[0].shape[2])) )
	responses = list()
	while idx < len(ims):
		sub_ims = ims[idx:idx+args.batch_size]
		for x, im in enumerate(sub_ims):
			network.blobs["data"].data[x,:,:,:] = im

		# propagate on batch
		network.forward()
		output = np.copy(network.blobs[output_blob].data)

		for x in xrange(len(sub_ims)):
			#responses[idx + x,0,:,:] = network.blobs[blob_name].data[x,0,:,:]
			responses.append(network.blobs[output_blob].data[x,0,:,:].copy())

		idx += args.batch_size
	return responses


LEFT_EDGE = -2
TOP_EDGE = -1
MIDDLE = 0
RIGHT_EDGE = 1
BOTTOM_EDGE = 2
# modified so that im is [channels, height, width]
def get_subwindows(im, pad_size, tile_size):
	height, width = tile_size, tile_size
	y_stride, x_stride = tile_size - (2 * pad_size), tile_size - (2 * pad_size)
	if (height > im.shape[1]) or (width > im.shape[2]):
		print "Invalid crop: crop dims larger than image (%r with %r)" % (im.shape, (height, width))
		exit(1)
	ims = list()
	locations = list()
	y = 0
	y_done = False
	while y  <= im.shape[1] and not y_done:
		x = 0
		if y + height > im.shape[1]:
			y = im.shape[1] - height
			y_done = True
		x_done = False
		while x <= im.shape[2] and not x_done:
			if x + width > im.shape[2]:
				x = im.shape[2] - width
				x_done = True
			locations.append( ((y, x, y + height, x + width), 
					(y + pad_size, x + pad_size, y + y_stride, x + x_stride),
					 TOP_EDGE if y == 0 else (BOTTOM_EDGE if y == (im.shape[1] - height) else MIDDLE),
					  LEFT_EDGE if x == 0 else (RIGHT_EDGE if x == (im.shape[2] - width) else MIDDLE) 
			) )
			ims.append(im[:,y:y+height,x:x+width])
			x += x_stride
		y += y_stride

	return locations, ims


def stich_together(locations, subwindows, size, dtype, pad_size, tile_size):
	output = np.zeros(size, dtype=dtype)
	for location, subwindow in zip(locations, subwindows):
		outer_bounding_box, inner_bounding_box, y_type, x_type = location
		y_paste, x_paste, y_cut, x_cut, height_paste, width_paste = -1, -1, -1, -1, -1, -1

		if y_type == TOP_EDGE:
			y_cut = 0
			y_paste = 0
			height_paste = tile_size - pad_size
		elif y_type == MIDDLE:
			y_cut = pad_size
			y_paste = inner_bounding_box[0]
			height_paste = tile_size - 2 * pad_size
		elif y_type == BOTTOM_EDGE:
			y_cut = pad_size
			y_paste = inner_bounding_box[0]
			height_paste = tile_size - pad_size

		if x_type == LEFT_EDGE:
			x_cut = 0
			x_paste = 0
			width_paste = tile_size - pad_size
		elif x_type == MIDDLE:
			x_cut = pad_size
			x_paste = inner_bounding_box[1]
			width_paste = tile_size - 2 * pad_size
		elif x_type == RIGHT_EDGE:
			x_cut = pad_size
			x_paste = inner_bounding_box[1]
			width_paste = tile_size - pad_size

		output[y_paste:y_paste+height_paste, x_paste:x_paste+width_paste] = subwindow[y_cut:y_cut+height_paste, x_cut:x_cut+width_paste]

	return output


def binarize(prob_map):
	out = np.zeros(prob_map.shape, np.uint8)
	high_indices = prob_map > 0.5
	out[high_indices] = 1
	return out


def update_predictions(net, data, args):
	print "Starting Predictions"
	for idx in xrange(len(data['original_images'])):
		im = data['original_images'][idx]
		locations, subwindows = get_subwindows(im, args.pad, args.tile_size)

		probs = predict(net, subwindows, 'probs', args)
		prob_map = stich_together(locations, probs, im.shape[1:], np.float32, args.pad, args.tile_size)
		data['probs'][idx] = prob_map
		data['binary'][idx] = binarize(prob_map)

		if idx and idx % args.print_count == 0:
			print "\tPredicted %d/%d" % (idx, len(data['original_images']))


num_errors = 0
def update_gt(data, args):
	print "Starting Adaptive GT"
	for idx in xrange(len(data['original_images'])):
		probs = data['probs'][idx]
		dists = data['baseline_dists'][idx]
		ep_dists = data['baseline_ep_dists'][idx]
		labels = data['baseline_labels'][idx]

		try:
			adapted_gt = adaptive_gt(probs, labels, dists, ep_dists, tolerance=args.tolerance, alpha=args.alpha)
		except:
			global num_errors
			adapted_gt = data['baselines'][idx].copy()
			if num_errors < 10:
				out_dir = os.path.join('errors', '%d' % num_errors)
				safe_mkdir(out_dir)

				np.save(os.path.join(out_dir, 'probs.npy'), probs)
				np.save(os.path.join(out_dir, 'labels.npy'), labels)
				np.save(os.path.join(out_dir, 'dists.npy'), dists)
				np.save(os.path.join(out_dir, 'ep_dists.npy'), ep_dists)
				open(os.path.join(out_dir, 'out.txt'), 'w').write("%s %d %f" % (data['filenames'][idx], args.tolerance, args.alpha))

			num_errors += 1

		if args.dilation_factor > 1:
			structure = np.ones( (args.dilation_factor, args.dilation_factor) )
			adapted_gt = nd.binary_dilation(adapted_gt, structure).astype(np.uint8)
		data['current_baselines'][idx] = adapted_gt

		if idx and idx % args.print_count == 0:
			print "\tAdapted %d/%d" % (idx, len(data['original_images']))


def load_data(manifest, _dir):
	dataset = collections.defaultdict(list)
	file_list = map(lambda s: s.strip(), open(manifest, 'r').readlines())
	for f in file_list:
		dataset['filenames'].append(f)
	for sub_dir in ['original_images', 'baselines']:
		for f in file_list:
			resolved = os.path.join(_dir, sub_dir, f)
			im = cv2.imread(resolved, int(sub_dir == 'original_images'))
			if im is None:
				raise Exception("Error loading %s" % resolved)
			dataset[sub_dir].append(im)
	for sub_dir in ['baseline_dists', 'baseline_labels', 'baseline_ep_dists']:
		for f in file_list:
			resolved = os.path.join(_dir, sub_dir, f + ".npy")
			im = np.load(resolved)
			if im is None:
				raise Exception("Error loading %s" % resolved)
			dataset[sub_dir].append(im)
	for f in file_list:
		dims_file = os.path.join(_dir, 'dims', f[:-4] + ".txt")
		original_size = open(dims_file, 'r').read().strip()
		tokens = original_size.split('x')

		# width/height as expected by cv2.resize
		dataset['original_size'].append( (int(tokens[0]), int(tokens[1])) )

	return dataset


def preprocess_data(data, args):
	for idx in xrange(len(data['original_images'])):
		im = data['original_images'][idx]
		im = args.scale * (im - args.mean)
		im = np.transpose(im, [2, 0, 1])
		data['original_images'][idx] = im

		gt = data['baselines'][idx]
		if args.dilation_factor > 1:
			structure = np.ones( (args.dilation_factor, args.dilation_factor) )
			gt = nd.binary_dilation(gt, structure).astype(np.uint8)
			data['baselines'][idx] = gt

		data['current_baselines'].append(gt.copy())
		data['probs'].append(np.zeros(gt.shape))
		data['binary'].append(np.zeros(gt.shape))


def get_solver_params(f):
	max_iters = 0
	snapshot = 0

	for line in open(f).readlines():
		tokens = line.split()
		if tokens[0] == 'max_iter:':
			max_iters = int(tokens[1])
		if tokens[0] == 'snapshot:':
			snapshot = int(tokens[1])
	return max_iters, snapshot


def presolve(net, args):
	net.blobs["data"].reshape(args.batch_size, 3, args.tile_size, args.tile_size)
	net.blobs["gt"].reshape(args.batch_size, 1, args.tile_size, args.tile_size)
	net.blobs["recall_weights"].reshape(args.batch_size, 1, args.tile_size, args.tile_size)
	net.blobs["precision_weights"].reshape(args.batch_size, 1, args.tile_size, args.tile_size)

	# fixed uniform weights for now
	net.blobs["recall_weights"].data[:] = np.ones( (args.batch_size, 1, args.tile_size, args.tile_size) )
	net.blobs["precision_weights"].data[:] = np.ones( (args.batch_size, 1, args.tile_size, args.tile_size) )


def sample_crop(shape, size):
	h, w = shape
	y = random.randint(0, h - size)
	x = random.randint(0, w - size)
	return y, x
	
	

def set_input_data(net, data, args):
	for batch_idx in xrange(args.batch_size):
		has_foreground = batch_idx > 0
		while not has_foreground:
			im_idx = random.randint(0, len(data['original_images']) - 1)
			im = data['original_images'][im_idx]
			gt = data['current_baselines'][im_idx]
			y, x = sample_crop(im.shape[1:], args.tile_size)

			im_crop = im[:,y:y+args.tile_size,x:x+args.tile_size]
			gt_crop = gt[y:y+args.tile_size,x:x+args.tile_size]
			if gt_crop.sum() > 0:
				has_foreground = True

		net.blobs["data"].data[batch_idx,:,:,:] = im_crop
		net.blobs["gt"].data[batch_idx,0,:,:] = gt_crop


def main(args):
	
	train_data = load_data(args.train_manifest, args.dataset_dir)
	val_data = load_data(args.val_manifest, args.dataset_dir)

	preprocess_data(train_data, args)
	preprocess_data(val_data, args)

	print "Done loading data"

	solver = caffe.SGDSolver(args.solver_file)
	max_iters, snapshot_interval = get_solver_params(args.solver_file)

	presolve(solver.net, args)

	for iter_num in xrange(max_iters):
		set_input_data(solver.net, train_data, args)
		solver.step(1)


		if iter_num and iter_num % snapshot_interval == 0:
			print "Validation Prediction"
			update_predictions(solver.net, val_data, args)
			if args.debug_dir:
				print "Dumping images"
				out_dir = os.path.join(args.debug_dir, 'val_%d' % iter_num)
				dump_debug(out_dir, val_data)

		if iter_num >= args.min_interval and iter_num % args.gt_interval == 0:
			print "Updating GT"

			update_predictions(solver.net, train_data, args)
			update_gt(train_data, args)
			if args.debug_dir:
				print "Dumping images"
				out_dir = os.path.join(args.debug_dir, 'train_%d' % iter_num)
				dump_debug(out_dir, train_data)
				
			

def get_args():
	parser = argparse.ArgumentParser(description="Outputs binary predictions")

	parser.add_argument("solver_file", 
				help="The solver.prototxt")
	parser.add_argument("dataset_dir",
				help="The dataset to be evaluated")
	parser.add_argument("train_manifest",
				help="txt file listing images to train on")
	parser.add_argument("val_manifest",
				help="txt file listing images for validation")

	parser.add_argument("--gpu", type=int, default=0,
				help="GPU to use for running the network")

	parser.add_argument("-m", "--mean", type=float, default=127.,
				help="Mean value for data preprocessing")
	parser.add_argument("-s", "--scale", type=float, default=0.0039,
				help="Optional pixel scale factor")
	parser.add_argument("-b", "--batch-size", default=2, type=int, 
				help="Training batch size")
	parser.add_argument("-d", "--dilation-factor", default=1, type=int, 
				help="Amount to dilate GT baselines")

	parser.add_argument("-p", "--pad", default=32, type=int, 
				help="Padding size for GT probability maps")
	parser.add_argument("-t", "--tile-size", default=256, type=int, 
				help="Size of tiles for training/prediction")
	parser.add_argument("--tolerance", default=5, type=int, 
				help="Size of tolerance regions around baselines")
	parser.add_argument("--alpha", default=0, type=float, 
				help="Coefficient for penalizing deviation from GT baseline")

	parser.add_argument("--gt-interval", default=500, type=int, 
				help="Interval for updating Adaptive GT")
	parser.add_argument("--min-interval", default=2000, type=int, 
				help="Miniumum iteration for updating Adaptive GT")

	parser.add_argument("--debug-dir", default='debug', type=str, 
				help="Dump images for debugging")
	parser.add_argument("--print-count", default=10, type=int, 
				help="Dump images for debugging")

	args = parser.parse_args()
	print args

	return args
			

if __name__ == "__main__":
	args = get_args()

	if args.gpu >= 0:
		caffe.set_device(args.gpu)
		caffe.set_mode_gpu()
	else:
		caffe.set_mode_cpu()

	main(args)

