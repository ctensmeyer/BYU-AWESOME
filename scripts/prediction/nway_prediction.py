#!/usr/bin/python

import os
import sys
import collections
import argparse
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import caffe
import cv2
import scipy.ndimage.morphology


LEFT_EDGE = -2
TOP_EDGE = -1
MIDDLE = 0
RIGHT_EDGE = 1
BOTTOM_EDGE = 2

def safe_mkdir(_dir):
	try:
		os.makedirs(_dir)
	except:
		pass


def setup_network(args):
	network = caffe.Net(args.net_file, args.weights_file, caffe.TEST)
	if args.gpu >= 0:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu)
	else:
		caffe.set_mode_cpu()
		
	return network


def fprop(network, ims, output_blobs, args):
	idx = 0

	responses = collections.defaultdict(list)
	while idx < len(ims):
		sub_ims = ims[idx:idx+args.batch_size]
		network.blobs["data"].reshape(len(sub_ims), ims[0].shape[2], ims[0].shape[0], ims[0].shape[1]) 
		for x, im in enumerate(sub_ims):
			transposed = np.transpose(im, [2,0,1])
			transposed = transposed[np.newaxis, :, :, :]
			network.blobs["data"].data[x,:,:,:] = transposed
		idx += args.batch_size

		# propagate on batch
		network.forward()
		for layer_name, blob_name in output_blobs:
			output = np.copy(network.blobs[blob_name].data)
			#print layer_name, blob_name, output.min(), output.max()
			responses[layer_name].append(output)

		print "Progress %d%%" % int(100 * min(idx, len(ims)) / float(len(ims)))
	#return np.concatenate(responses, axis=0)
	return {key: np.concatenate(value, axis=0) for key, value in responses.iteritems()}


def predict(network, ims, output_blobs, args):
	raw_outputs = fprop(network, ims, output_blobs, args)
	thresholded_outputs = dict()
	for key, raw_output in raw_outputs.iteritems():
		thresholded_outputs[key] = np.argmax(raw_output, axis=1)
	return thresholded_outputs


def get_subwindows(im, pad_size, tile_size):
	height, width = tile_size, tile_size
	y_stride, x_stride = tile_size - (2 * pad_size), tile_size - (2 * pad_size)
	if (height > im.shape[0]) or (width > im.shape[1]):
		print "Invalid crop: crop dims larger than image (%r with %r)" % (im.shape, (height, width))
		exit(1)
	ims = list()
	locations = list()
	y = 0
	y_done = False
	while y  <= im.shape[0] and not y_done:
		x = 0
		if y + height > im.shape[0]:
			y = im.shape[0] - height
			y_done = True
		x_done = False
		while x <= im.shape[1] and not x_done:
			if x + width > im.shape[1]:
				x = im.shape[1] - width
				x_done = True
			locations.append( ((y, x, y + height, x + width), 
					(y + pad_size, x + pad_size, y + y_stride, x + x_stride),
					 TOP_EDGE if y == 0 else (BOTTOM_EDGE if y == (im.shape[0] - height) else MIDDLE),
					  LEFT_EDGE if x == 0 else (RIGHT_EDGE if x == (im.shape[1] - width) else MIDDLE) 
			) )
			ims.append(im[y:y+height,x:x+width])
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


def save_histo(data, fname, title, weights=None):
	if weights is not None:
		weights = weights.flatten()
		
	n, bins, patches = plt.hist(data.flatten(), bins=100, weights=weights, log=True)
	plt.title(title)
	plt.xlabel('Predicted Probability of Foreground')
	plt.ylabel('Pixel Count')
	plt.tick_params(axis='y', which='minor', left='off', right='off')

	plt.savefig(fname)
	plt.clf()


def xor_image(im1, im2):
	out_image = np.zeros(im1.shape + (3,))

	for y in xrange(im1.shape[0]):
	    for x in xrange(im1.shape[1]):
			if im1[y,x]:
				if im2[y,x]:
					# white on white
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
					out_image[y,x] = (0,0,0)
	return out_image


def get_ims_files(args):
	im_files = map(lambda s: s.strip(), open(args.image_manifest, 'r').readlines())
	im_dirs = args.im_dirs.split(',')
	return im_files, im_dirs


def load_im(im_file, im_dirs, args):
	ims = list()
	for im_dir in im_dirs:
		im_path = os.path.join(args.dataset_dir, im_dir, im_file)
		im = cv2.imread(im_path, -1)  # reads in image as is
		if im is None:
			raise Exception("File does not exist: %s" % im_path)
		if im.ndim == 2:
			im = im[:,:,np.newaxis]
		ims.append(im)
	im = np.concatenate(ims, axis=2)
	im = im - args.mean
	im = args.scale * im
	return im


def write_output(locations, nway_subwindows, im_file, image, pad_size, im, args):
	for key in nway_subwindows.keys():
		result = stich_together(locations, nway_subwindows[key], tuple(image.shape[0:2]), 
			np.uint8, pad_size, args.tile_size)
		out_file = os.path.join(args.out_dir, 'nway', key, im_file)
		cv2.imwrite(out_file, result)

		colored = colormap(result)
		out_file = os.path.join(args.out_dir, 'colored', key, im_file)
		cv2.imwrite(out_file, colored)

colors = [ (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255),
		   (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0),
		   (0, 255, 128), (128, 255, 0)]
def colormap(im):
	out_im = np.zeros( (im.shape[0], im.shape[1], 3), dtype=np.uint8 )
	max_val = np.max(im)
	for val in xrange(max_val + 1):
		indices = (im == val)
		out_im[indices] = colors[val % len(colors)]
	return out_im

def get_output_blobs(f):
	output_blobs = list()
	lines = open(f).readlines()
	for idx, line in enumerate(lines):
		if 'Softmax' in line:
			# layer name is meaningful
			layer_name = lines[idx-1].split()[-1].strip('"')

			# top name is autogenerated, but what the network recognizes
			top_name = lines[idx+2].split()[-1].strip('"')
			output_blobs.append( (layer_name, top_name) )
	return output_blobs


def main(args):
	output_blobs = get_output_blobs(args.net_file)
	for layer_name, _ in output_blobs:
		safe_mkdir(os.path.join(args.out_dir, 'colored', layer_name))
		safe_mkdir(os.path.join(args.out_dir, 'nway', layer_name))
	network = setup_network(args)
	im_files, im_dirs = get_ims_files(args)
	pad_size = args.pad
	for idx, im_file in enumerate(im_files):
		image = load_im(im_file, im_dirs, args)
		if idx == 0:
			print image.shape
		if idx and idx % args.print_count == 0:
			print "Processed %d/%d Images" % (idx, len(im_files))
		locations, subwindows = get_subwindows(image, pad_size, args.tile_size)
		nway_subwindows = predict(network, subwindows, output_blobs, args)
		write_output(locations, nway_subwindows, im_file, image, pad_size, image, args)

	

def get_args():
	parser = argparse.ArgumentParser(description="Outputs binary predictions")

	parser.add_argument("net_file", 
				help="The deploy.prototxt")
	parser.add_argument("weights_file", 
				help="The weights.caffemodel")
	parser.add_argument("dataset_dir",
				help="The dataset to be evaluated")
	parser.add_argument("image_manifest",
				help="txt file listing images to evaluate")
	parser.add_argument("out_dir",
				help="output directory")

	parser.add_argument("--gpu", type=int, default=0,
				help="GPU to use for running the network")

	parser.add_argument("-m", "--mean", type=float, default=127.,
				help="Mean value for data preprocessing")
	parser.add_argument("-a", "--scale", type=float, default=0.0039,
				help="Optional scale factor")
	parser.add_argument("--print-count", default=1, type=int, 
				help="Print every print-count images processed")
	parser.add_argument("-b", "--batch-size", default=4, type=int, 
				help="Max number of transforms in single batch per original image")
	parser.add_argument("-p", "--pad", default=0, type=int, 
				help="Padding size")
	parser.add_argument("-t", "--tile-size", default=384, type=int, 
				help="Size of tiles to extract")
	parser.add_argument("--im-dirs", default='original_images', type=str, 
				help="comma separated list of input images to the network")

	args = parser.parse_args()
	print args

	return args
			

if __name__ == "__main__":
	args = get_args()
	main(args)

