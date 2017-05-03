import os
import sys
import argparse
import lmdb
import StringIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import caffe.proto.caffe_pb2
import traceback
import numpy as np
import cv2
import shutil
import math

def process_im(im_file):
	im = cv2.imread(im_file, cv2.IMREAD_UNCHANGED)
	return im


def open_db(db_file):
	env = lmdb.open(db_file, readonly=False, map_size=int(2 ** 38), writemap=False, max_readers=10000)
	txn = env.begin(write=True)
	return env, txn
	

def package(im, encoding):
	doc_datum = caffe.proto.caffe_pb2.DocumentDatum()
	datum_im = doc_datum.image

	datum_im.channels = im.shape[2] if len(im.shape) == 3 else 1
	datum_im.width = im.shape[1]
	datum_im.height = im.shape[0]
	datum_im.encoding = args.encoding

	# image data
	if encoding != 'none':
		buf = StringIO.StringIO()
		if datum_im.channels == 1:
			plt.imsave(buf, im, format=encoding, vmin=0, vmax=255, cmap='gray')
		else:
			plt.imsave(buf, im, format=encoding, vmin=0, vmax=1)
		datum_im.data = buf.getvalue()
	else:
		pix = im.transpose(2, 0, 1)
		datum_im.data = pix.tostring()

	return doc_datum


def split_im(im, tile_size):
	ims = list()
	height, width = im.shape[:2]

	if height < tile_size:
		diff = tile_size - height
		pad_arg = ( (0, diff), (0, 0) )
		if im.ndim == 3:
			pad_arg += (0, 0),
		im = np.pad(im, pad_arg, mode='reflect')

	if width < tile_size:
		diff = tile_size - width
		pad_arg = ( (0, 0), (0, width) )
		if im.ndim == 3:
			pad_arg += (0, 0),
		im = np.pad(im, pad_arg, mode='reflect')
		
	height, width = im.shape[:2]
	height_splits = math.ceil(height / float(tile_size))
	width_splits = math.ceil(width / float(tile_size))

	height_overlap = tile_size - (height / float(height_splits))
	if height_overlap < tile_size / 10.:
		height_splits += 1
	width_overlap = tile_size - (width / float(width_splits))
	if width_overlap < tile_size / 10.:
		width_splits += 1

	height_splits = int(height_splits)
	width_splits = int(width_splits)
	height_overlap = tile_size - (height / height_splits)
	width_overlap = tile_size - (width / width_splits)

	height_stride = (height - tile_size) / (height_splits - 1) if height_splits > 1 else 0
	width_stride = (width - tile_size) / (width_splits - 1) if width_splits > 1 else 0

	for y in xrange(height_splits):
		h_start = y * height_stride
		for x in xrange(width_splits):
			w_start = x * width_stride
			if im.ndim == 2:
				tile = im[h_start:h_start+tile_size,w_start:w_start+tile_size]
			else:
				tile = im[h_start:h_start+tile_size,w_start:w_start+tile_size,:]
			if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
				ims.append(tile)

	return ims



def create_lmdb(imroot, imlist, db_file, encoding, tile_size, gtroot, remove_background):
	print "Starting on %s" % db_file
	env, txn = open_db(db_file)
	for x, imname in enumerate(imlist):
		if x and x % 10 == 0:
			print "Processed %d images" % x
		try:
			im_file = os.path.join(imroot, imname)
			gt_file = os.path.join(gtroot, imname)
			im = process_im(im_file)
			gt = process_im(gt_file)
			if x == 0:
				print im.shape
			ims = split_im(im, tile_size)
			gts = split_im(gt, tile_size)
			assert all(map(lambda x,y: x.shape[:2] == y.shape[:2], ims, gts))

			# remove patches containing all background
			if remove_background:
				idx = 0
				while idx < len(ims):
					gt = gts[idx]
					if gt.max() == 0:
						#print "Deleting patch %d of image %s" % (idx, im_file)
						del gts[idx]
						del ims[idx]
					else:
						idx += 1
			#for idx, im in enumerate(ims):
			#	cv2.imwrite("tmp/%d.png" % idx, im)
			#exit()

			for idx, im in enumerate(ims):
				doc_datum = package(im, encoding)

				key = "%d:%d:%s" % (idx * 76547000 + x * 37, x, os.path.splitext(os.path.basename(im_file))[0])
				txn.put(key, doc_datum.SerializeToString())
			if x % 10 == 0:
				txn.commit()
				env.sync()
				print env.stat()
				print env.info()
				txn = env.begin(write=True)

		except Exception as e:
			print e
			print traceback.print_exc(file=sys.stdout)
			print "Error occured on:", im_file
			raise


	print "Done Processing Images"
	txn.commit()
	env.close()


def readfile(fn):
	with open(fn, 'r') as fd:
		lines = fd.readlines()
		return map(lambda s: s.rstrip(), lines)


# TODO: changed to tiles, so #images != #entries
def check_lmdb(lmdb_dir, expected_num_entries):
	if not os.path.exists(lmdb_dir):
		return False
	#env = lmdb.open(lmdb_dir, readonly=True, map_size=int(2 ** 38))
	#num_entries = env.stat()['entries']
	#delete = num_entries != expected_num_entries
	#env.close()

	#if delete:
	#	shutil.rmtree(lmdb_dir)
	#	return False
	return True


def main(args):
	in_dir = args.in_dir
	out_root = os.path.join(in_dir, 'lmdb', str(args.size))
	try:
		os.makedirs(out_root)
	except:
		pass
	labels_dir = os.path.join(in_dir, 'labels')
	labels_files = []
	for f in os.listdir(labels_dir):
		if any(x in f for x in ('train', 'val', 'test')):
			labels_files.append(os.path.join(labels_dir, f))

	gtroot = os.path.join(in_dir, args.gt_name)
	for label_file in labels_files:
		imnames = readfile(label_file)
		num_files = len(imnames)
		label_type = os.path.splitext(os.path.basename(label_file))[0]
		for d in os.listdir(in_dir):
			if any(x in d for x in ('labels', 'tmp', 'lmdb', 'pr_dats')):
				continue
			rd = os.path.join(in_dir, d)
			fns = os.listdir(rd)

			if any(os.path.isfile(os.path.join(rd,fn)) for fn in fns):
				# contains images
				imroot = rd
				out_dir = os.path.join(out_root, d)
				try:
					os.makedirs(out_dir)
				except:
					pass

				out_lmdb = os.path.join(out_dir,  "%s_%s_lmdb" % (d, label_type))
				if check_lmdb(out_lmdb, num_files):
					print "Skipping %s, already done" % out_lmdb
				else:
					create_lmdb(imroot, imnames, out_lmdb, args.encoding, args.size, gtroot, args.remove_background)
			else:
				# process each sub directory
				for sd in os.listdir(rd):
					rsd = os.path.join(rd, sd)
					if any(os.path.isfile(os.path.join(rsd,fn)) for fn in os.listdir(rsd)):
						out_dir = os.path.join(out_root, d)
						try:
							os.makedirs(out_dir)
						except:
							pass
						imroot = rsd

						out_lmdb = os.path.join(out_dir, "%s_%s_%s_lmdb" % (d, sd, label_type)) 
						if check_lmdb(out_lmdb, num_files):
							print "Skipping %s, already done" % out_lmdb
						else:
							create_lmdb(imroot, imnames, out_lmdb, args.encoding, args.size, gtroot, args.remove_background)

					else:
						# relative darkness/canny
						out_dir = os.path.join(out_root, d, sd)
						try:
							os.makedirs(out_dir)
						except:
							pass
						for ssd in os.listdir(rsd):
							rssd = os.path.join(rsd, ssd)
							if any(os.path.isfile(os.path.join(rssd,fn)) for fn in os.listdir(rssd)):
								imroot = rssd

								out_lmdb = os.path.join(out_dir, "%s_%s_%s_%s_lmdb" % (d, sd, ssd, label_type)) 
								if check_lmdb(out_lmdb, num_files):
									print "Skipping %s, already done" % out_lmdb
								else:
									create_lmdb(imroot, imnames, out_lmdb, args.encoding, args.size, gtroot, args.remove_background)
							else:
								# relative_darkness2
								out_dir = os.path.join(out_root, d, sd, ssd)
								try:
									os.makedirs(out_dir)
								except:
									pass
								for sssd in os.listdir(rssd):
									rsssd = os.path.join(rssd, sssd)
									imroot = rsssd

									out_lmdb = os.path.join(out_dir, "%s_%s_%s_%s_%s_lmdb" % (d, sd, ssd, sssd, label_type)) 
									if check_lmdb(out_lmdb, num_files):
										print "Skipping %s, already done" % out_lmdb
									else:
										create_lmdb(imroot, imnames, out_lmdb, args.encoding, args.size, gtroot, args.remove_background)



def get_args():
	parser = argparse.ArgumentParser(description="Creates an LMDB of DocumentDatums")
	parser.add_argument('in_dir', type=str,
						help='Directory to process')

	parser.add_argument('-e', '--encoding', type=str, default='png',
						help='How to store the image in the DocumentDatum')
	parser.add_argument('-s', '--size', type=int, default=256,
						help='Size to tile each image')
	parser.add_argument('--remove-background', default=False, action='store_true',
						help='Whether to only extract patches that have some positive GT pixels')
	parser.add_argument('--gt-name', default='baselines_1', type=str,
						help='Name of directory of the GT for determining backgorund for remove-background=True')

	args = parser.parse_args()
	return args



if __name__ == "__main__":
	args = get_args();
	print "%s started with..." % __file__
	print args
	main(args)
	#im = cv2.imread(sys.argv[1])
	#ims = split_im(im, int(sys.argv[2]))
	#for x, im in enumerate(ims):
	#	cv2.imwrite("/home/chris/Dropbox/tmp/%d.png" % x, im)

