#!/usr/bin/python

import argparse
import os
import re

import make_transforms

import caffe
from caffe import layers as L
from caffe import params as P
import caffe.proto.caffe_pb2 as proto

import numpy as np
import random
import collections
import lmdb

ROOT="/fslgroup/fslg_icdar/compute"

SIZES=[512,384,256,150,100,64,32]

OUTPUT_SIZES = {"andoc_1m": 974, "andoc_1m_10": 974, "andoc_1m_50": 974, "andoc_1m_half": 974, "rvl_cdip": 16, "rvl_cdip_10": 16, "rvl_cdip_100": 16, "rvl_cdip_half": 16, "imagenet": 1000, "combined": (974 + 16)}

MEAN_VALUES = { "andoc_1m": {"hsv": [16, 22, 178], "binary": [194], "binary_invert": [61], "gray": [175], "gray_invert": [80], "gray_padded": [None], "color": [178,175,166], "color_invert": [77,80,89], "color_padded": [126,124,118], "color_multiple": [178,175,166], "dsurf": [128], "color_short": [178,175,166]},
				"rvl_cdip": {"binary": [233], "binary_invert": [22], "gray": [234], "color": [234, 234, 234], "color_invert": [21, 21, 21], "gray_padded": [239], "gray_invert": [21], "gray_multiple": [234], "dsurf": [128], "gray_short": [234]},
				"imagenet": {"color": [104,117,123], "gray": [114]},
				"combined": {"color": [194, 191, 185], "gray": [191]}
			  }
MEAN_VALUES['rvl_cdip_10'] = MEAN_VALUES['rvl_cdip']
MEAN_VALUES['rvl_cdip_100'] = MEAN_VALUES['rvl_cdip']
MEAN_VALUES['rvl_cdip_half'] = MEAN_VALUES['rvl_cdip']
MEAN_VALUES['andoc_1m_10'] = MEAN_VALUES['andoc_1m']
MEAN_VALUES['andoc_1m_50'] = MEAN_VALUES['andoc_1m']
MEAN_VALUES['andoc_1m_half'] = MEAN_VALUES['andoc_1m']

#DEFAULT_TEST_TRANSFORMS = [10]
DEFAULT_TEST_TRANSFORMS = []


def get_num_channels(lmdbs):
	count = 0
	for lmdb in lmdbs:
		if 'gray' in lmdb:
			if 'relative' in lmdb:
				count += 3
			else:
				count += 1
		else:
			count += 3
	return count

def lmdb_num_entries(db_path):
	env = lmdb.open(db_path, readonly=True)
	stats = env.stat()
	num_entries = stats['entries']
	env.close()
	return num_entries
	

def OUTPUT_FOLDER(dataset, group, experiment, split):
	return os.path.join("experiments/preprocessing/nets" , dataset, group, experiment, split)


def OUTPUT_FOLDER_BINARIZE(dataset, group, experiment, split):
	return os.path.join("experiments/binarize/nets" , dataset, group, experiment, split)


def TRANSFORMS_FOLDER(dataset, group, experiment, split):
	return os.path.join(OUTPUT_FOLDER(dataset,group,experiment,split), "transforms")


def EXPERIMENTS_FOLDER(dataset, group, experiment, split):
	return os.path.join(ROOT, OUTPUT_FOLDER(dataset, group, experiment, split))


def EXPERIMENTS_FOLDER_BINARIZE(dataset, group, experiment, split):
	return os.path.join(ROOT, OUTPUT_FOLDER_BINARIZE(dataset, group, experiment, split))


def LMDB_MULTIPLE_PATH(dataset, tag, split):
	lmdbs = collections.defaultdict(list)
	for s in 'train_lmdb', 'val_lmdb', 'test_lmdb':
		par_dir = os.path.join(ROOT, "lmdb", dataset, tag, split, s)
		for s_dir in os.listdir(par_dir):
			r_dir = os.path.join(par_dir, s_dir)
			lmdbs[s].append(r_dir)
		
		# use the same val/test lmdbs for the smaller versions of rvl_cdip and andoc_1m
		if dataset.startswith('rvl_cdip'):
			dataset = 'rvl_cdip'
		if dataset.startswith('andoc_1m'):
			dataset = 'andoc_1m'

	return lmdbs['train_lmdb'], lmdbs['val_lmdb'], lmdbs['test_lmdb']
	

def LMDB_PATH(dataset, tag, split):
	#return map(lambda s: os.path.join(ROOT, "lmdb", dataset, tag, split, s), ["train_lmdb", "val_lmdb", "test_lmdb"])
	lmdbs = list()
	lmdbs.append(os.path.join(ROOT, "lmdb", dataset, tag, split, "train_lmdb"))
	if dataset.startswith('rvl_cdip'):
		dataset = 'rvl_cdip'
	if dataset.startswith('andoc_1m'):
		dataset = 'andoc_1m'
	for s in ['val_lmdb', 'test_lmdb']:
		lmdbs.append(os.path.join(ROOT, "lmdb", dataset, tag, split, s))
	return lmdbs


def LMDB_PATH_BINARIZE(dataset, tag, size, data_partition='train'):
	if not isinstance(tag, basestring):
		path = os.path.join(ROOT, "data/chris", dataset, 'lmdb', str(size), tag[0], "%s_%s_lmdb" % (tag[1], data_partition))
		#path = os.path.join(ROOT, "lmdb", dataset, str(size), tag[0], "%s_%s_lmdb" % (tag[1], data_partition))
		#path = os.path.join(ROOT, "lmdb", dataset, tag[0], "%s_%s_lmdb" % (tag[1], data_partition))
	else:
		path = os.path.join(ROOT, "data/chris", dataset, 'lmdb', str(size), tag, "%s_%s_lmdb" % (tag, data_partition))
		#path = os.path.join(ROOT, "lmdb", dataset, tag, "%s_%s_lmdb" % (tag, data_partition))
	return path
	

def getSizeFromTag(t):
	tokens = t.split("_")
	return int(tokens[1])
	#return map(int, re.sub("(_?[^0-9_]+_?)","", t).split("_"))


def getTagWithoutSize(t):
	return re.sub("_*[0-9]+","", t)


def getNumChannels(tags):
	channels = 0

	for t in tags:
		if "color" in t:
			channels += 3
		elif "gray" in t:
			channels += 1
		elif "binary" in t:
			channels += 1

	return channels


def poolLayer(prev, bn=False, **kwargs):
	return L.Pooling(prev, pool=P.Pooling.MAX, **kwargs)


def convLayer(prev, lrn=False, param_name=None, bn=False, **kwargs):
	if param_name:
		name1 = param_name + '_kernels'
		name2 = param_name + '_bias'
		conv = L.Convolution(prev, param=[dict(lr_mult=1, name=name1), dict(lr_mult=2, name=name2)], 
			weight_filler=dict(type='msra'), **kwargs)
	else:
		conv = L.Convolution(prev, param=[dict(lr_mult=1), dict(lr_mult=2)], 
			weight_filler=dict(type='msra'), **kwargs)
	if bn:
		bn = L.BatchNorm(conv)
		relu = L.ReLU(bn, in_place=True)
	else:
		relu = L.ReLU(conv, in_place=True)
	if lrn:
		# optional Local Response Normalization
		relu = L.LRN(relu, lrn_param={'local_size': min(kwargs['num_output'] / 3, 5), 'alpha': 0.0001, 'beta': 0.75})
	return relu


def convLayerSigmoid(prev, bn=False, **kwargs):
	conv = L.Convolution(prev, param=[dict(lr_mult=1), dict(lr_mult=2)], weight_filler=dict(type='msra'), **kwargs)
	sigmoid = L.Sigmoid(conv, in_place=True)
	return sigmoid


def convLayerOnly(prev, bn=False, **kwargs):
	conv = L.Convolution(prev, param=[dict(lr_mult=1), dict(lr_mult=2)], weight_filler=dict(type='msra'), **kwargs)
	return conv


def ipLayer(prev, param_name=None, bn=False, **kwargs):
	if param_name:
		name1 = param_name + '_weights'
		name2 = param_name + '_bias'
		return L.InnerProduct(prev, 
			param=[dict(lr_mult=1, name=name1), dict(lr_mult=2, name=name2)], 
			weight_filler=dict(type='msra'), bias_filler=dict(type='constant'), **kwargs) 
	else:
		return L.InnerProduct(prev, 
			param=[dict(lr_mult=1), dict(lr_mult=2)], 
			weight_filler=dict(type='msra'), bias_filler=dict(type='constant'), **kwargs) 


def fcLayer(prev, bn=False, **kwargs):
	fc = ipLayer(prev, **kwargs)
	if bn:
		bn = L.BatchNorm(fc)
		relu = L.ReLU(bn, in_place=True)
	else:
		relu = L.ReLU(fc, in_place=True)
	return relu


# for use in equivariance experiments
BEFORE_LAYERS = [(convLayer, {"name": "conv1", "kernel_size": 11, "num_output": 96, "stride": 4, "param_name": "conv1" }), 
		  (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
		  (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
		  (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":256, "pad": 2, "param_name": "conv2" }),
		  (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
		  (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
		  (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":384, "pad": 1, "param_name": "conv3" }),
		  (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":384, "pad": 1, "param_name": "conv4" }),
		  (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":256, "pad": 1, "param_name": "conv5" }),
		  (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2}),
		  (fcLayer, {"name": "fc6", "num_output": 4096, "param_name": "fc6" }),
		  (L.Dropout, {"name": "dropout6", "dropout_ratio": 0.5, "in_place": True}),
		  (ipLayer, {"name": "fc7", "num_output": 4096, "param_name": "fc7" }),
		  (L.ReLU, {"name": "fc7-relu", "in_place": False}),
		]


AFTER_LAYERS = [
		  (L.Dropout, {"name": "dropout7", "dropout_ratio": 0.5, "in_place": False})
		]
	
DEPTH_FC_LAYERS = { 
			  1 : [(fcLayer, {"name": "fc6", "num_output": 4096}),
				   (L.Dropout, {"name": "dropout6", "dropout_ratio": 0.5, "in_place": True})],
			  2 : [(fcLayer, {"name": "fc6", "num_output": 4096}),
				   (L.Dropout, {"name": "dropout6", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc7", "num_output": 4096}),
				   (L.Dropout, {"name": "dropout7", "dropout_ratio": 0.5, "in_place": True})],
			  3 : [(fcLayer, {"name": "fc6", "num_output": 4096}),
				   (L.Dropout, {"name": "dropout6", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc7", "num_output": 4096}),
				   (L.Dropout, {"name": "dropout7", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc8", "num_output": 4096}),
				   (L.Dropout, {"name": "dropout8", "dropout_ratio": 0.5, "in_place": True})],
			  4 : [(fcLayer, {"name": "fc6", "num_output": 4096}),
				   (L.Dropout, {"name": "dropout6", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc7", "num_output": 4096}),
				   (L.Dropout, {"name": "dropout7", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc8", "num_output": 4096}),
				   (L.Dropout, {"name": "dropout8", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc9", "num_output": 4096}),
				   (L.Dropout, {"name": "dropout9", "dropout_ratio": 0.5, "in_place": True})],
			}

# all sized for 227x227
DEPTH_CONV_LAYERS = { 0 : [(convLayer, {"name": "conv1", "kernel_size": 11, "num_output": 96, "stride": 4}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":256, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],
				 1 : [(convLayer, {"name": "conv1", "kernel_size": 11, "num_output": 96, "stride": 4}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":256, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":256, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],
				 2 : [(convLayer, {"name": "conv1", "kernel_size": 11, "num_output": 96, "stride": 4}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":256, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":256, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],
				 3 : [(convLayer, {"name": "conv1", "kernel_size": 11, "num_output": 96, "stride": 4}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":256, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":256, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],
				 4 : [(convLayer, {"name": "conv1", "kernel_size": 11, "num_output": 96, "stride": 4}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":256, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv6", "kernel_size": 3, "num_output":256, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],
	
				 5 : [(convLayer, {"name": "conv1", "kernel_size": 11, "num_output": 96, "stride": 4}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":256, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv6", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv7", "kernel_size": 3, "num_output":256, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],

				 6 : [(convLayer, {"name": "conv1", "kernel_size": 11, "num_output": 96, "stride": 4}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":256, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv6", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv7", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv8", "kernel_size": 3, "num_output":256, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					]
				}
				
CONV_LAYERS = {
			   32:  [(convLayer, {"name": "conv1", "kernel_size": 5, "num_output": 24, "stride": 1}), 
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":64, "pad": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":96, "pad": 1}),
					 (poolLayer, {"name": "pool3", "kernel_size": 3, "stride": 2}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":96, "pad": 0}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":64, "pad": 0}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],

			   64:  [(convLayer, {"name": "conv1", "kernel_size": 7, "num_output": 32, "stride": 1}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":96, "pad": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":148, "pad": 1}),
					 (poolLayer, {"name": "pool3", "kernel_size": 3, "stride": 2}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":148, "pad": 0}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":96, "pad": 0}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],

			   100: [(convLayer, {"name": "conv1", "kernel_size": 9, "num_output": 48, "stride": 2}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":128, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":192, "pad": 1}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":192, "pad": 1}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":128, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],

			   150: [(convLayer, {"name": "conv1", "kernel_size": 11, "num_output": 64, "stride": 3}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":192, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":256, "pad": 1}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":256, "pad": 1}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":192, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],


			   227: [(convLayer, {"name": "conv1", "kernel_size": 11, "num_output": 96, "stride": 4}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":256, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":256, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],

			   256: [(convLayer, {"name": "conv1", "kernel_size": 11, "num_output": 96, "stride": 4}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":256, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":384, "pad": 0}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":256, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],

			   320: [(convLayer, {"name": "conv1", "kernel_size": 15, "num_output": 96, "stride": 5}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":256, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":384, "pad": 0}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":256, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],

			   384: [(convLayer, {"name": "conv1", "kernel_size": 15, "num_output": 120, "stride": 3}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 7, "num_output":320, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 5, "num_output":448, "pad": 1}),
					 (poolLayer, {"name": "pool3", "kernel_size": 3, "stride": 2}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":448, "pad": 0}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":320, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],

			   512: [(convLayer, {"name": "conv1", "kernel_size": 15, "num_output": 144, "stride": 4}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 7, "num_output":384, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 5, "num_output":512, "pad": 1}),
					 (poolLayer, {"name": "pool3", "kernel_size": 3, "stride": 2}),
					 (convLayer, {"name": "conv4", "kernel_size": 5, "num_output":512, "pad": 1}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],
			  }



FC_LAYERS = {

			 32:  [(fcLayer, {"name": "fc6", "num_output": 1024}),
				   (L.Dropout, {"name": "dropout6", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc7", "num_output": 1024}),
				   (L.Dropout, {"name": "dropout7", "dropout_ratio": 0.5, "in_place": True})],

			 64:  [(fcLayer, {"name": "fc6", "num_output": 1536}),
				   (L.Dropout, {"name": "dropout6", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc7", "num_output": 1536}),
				   (L.Dropout, {"name": "dropout7", "dropout_ratio": 0.5, "in_place": True})],

			 100: [(fcLayer, {"name": "fc6", "num_output": 2048}),
				   (L.Dropout, {"name": "dropout6", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc7", "num_output": 2048}),
				   (L.Dropout, {"name": "dropout7", "dropout_ratio": 0.5, "in_place": True})],

			 150: [(fcLayer, {"name": "fc6", "num_output": 3072}),
				   (L.Dropout, {"name": "dropout6", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc7", "num_output": 3072}),
				   (L.Dropout, {"name": "dropout7", "dropout_ratio": 0.5, "in_place": True})],

			 227: [(fcLayer, {"name": "fc6", "num_output": 4096}),
				   (L.Dropout, {"name": "dropout6", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc7", "num_output": 4096}),
				   (L.Dropout, {"name": "dropout7", "dropout_ratio": 0.5, "in_place": True})],

			 256: [(fcLayer, {"name": "fc6", "num_output": 4096}),
				   (L.Dropout, {"name": "dropout6", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc7", "num_output": 4096}),
				   (L.Dropout, {"name": "dropout7", "dropout_ratio": 0.5, "in_place": True})],

			 320: [(fcLayer, {"name": "fc6", "num_output": 4096}),
				   (L.Dropout, {"name": "dropout6", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc7", "num_output": 4096}),
				   (L.Dropout, {"name": "dropout7", "dropout_ratio": 0.5, "in_place": True})],
				   
			 384: [(fcLayer, {"name": "fc6", "num_output": 5120}),
				   (L.Dropout, {"name": "dropout6", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc7", "num_output": 5120}),
				   (L.Dropout, {"name": "dropout7", "dropout_ratio": 0.5, "in_place": True})],

			 512: [(fcLayer, {"name": "fc6", "num_output": 6144}),
				   (L.Dropout, {"name": "dropout6", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc7", "num_output": 6144}),
				   (L.Dropout, {"name": "dropout7", "dropout_ratio": 0.5, "in_place": True})],
				   
			}



VAL_BATCH_SIZE = 40
TRAIN_TRAIN = "train_train.prototxt"
TRAIN_VAL = "train_val.prototxt"
TRAIN_TEST = "train_test.prototxt"
TRAIN_TEST2 = "train_test2.prototxt"
DEPLOY_FILE = "deploy.prototxt"
SOLVER = "solver.prototxt"
SNAPSHOT_FOLDER = "snapshots"

LEARNING_RATES = {"combined": 0.005, "andoc_1m": 0.005, "rvl_cdip": 0.003, "imagenet": 0.01}
BATCH_SIZE = {"combined": 128, "andoc_1m": 128, "rvl_cdip": 32, "imagenet": 256}
MAX_ITER = {"combined": 350000, "andoc_1m": 250000, "rvl_cdip": 500000, "imagenet": 450000}
STEP_SIZE = {"combined": 150000, "andoc_1m": 100000, "rvl_cdip": 150000, "imagenet": 100000}

for d in [LEARNING_RATES, BATCH_SIZE, MAX_ITER, STEP_SIZE]:
	d['rvl_cdip_10'] = d['rvl_cdip']
	d['rvl_cdip_100'] = d['rvl_cdip']
	d['rvl_cdip_half'] = d['rvl_cdip']
	d['andoc_1m_10'] = d['andoc_1m']
	d['andoc_1m_50'] = d['andoc_1m']
	d['andoc_1m_half'] = d['andoc_1m']


MAX_ITER['rvl_cdip_100'] = 150000
STEP_SIZE['rvl_cdip_100'] = 50000

SOLVER_PARAM = {#"test_iter": 1000, 
				"test_interval": 5000, 
				"lr_policy": '"step"',
				"gamma": 0.1,
				"display": 100,
				"momentum": 0.9,
				"weight_decay": 0.0005,
				"snapshot": 5000,
				"solver_mode": "GPU"}

# this is for validation sets, not test sets.  Test set iterations are specified in train.sh
MULTIPLE_TEST_ITERS  = { "andoc_1m" : { "color_227_multiple": 1007, "color_384_multiple": 1013, 
										"color_227_multiple2": 1005, "color_384_multiple2": 1005},
						 "rvl_cdip" : { "gray_227_multiple":  1006, "gray_384_multiple": 1008, 
						 				"gray_227_multiple2": 1005, "gray_384_multiple2": 1005}
					   }

def createLinearParam(shift=0.0, scale=1.0, **kwargs):
	return dict(shift=shift, scale=scale)

def createColorJitterParam(sigma=5):
	return dict(sigma=sigma)

def createColorJitterParamE(params):
	return dict(mean=params[0], sigma=params[1])

def createCropParam(phase):
	if phase == caffe.TRAIN:
		location = P.CropTransform.RANDOM
	else:
		location = P.CropTransform.CENTER

	return dict(size=227, location=location)

def createCropParamE(params):
	location = {'center': P.CropTransform.CENTER,
				'random': P.CropTransform.RANDOM,
				'rand_corner': P.CropTransform.RAND_CORNER,
				'ul': P.CropTransform.UL_CORNER,
				'ur': P.CropTransform.UR_CORNER,
				'bl': P.CropTransform.BL_CORNER,
				'br': P.CropTransform.BR_CORNER,
		}[params[0].lower()]
	return dict(size=227, location=location)

def createReflectParam(hmirror=0.0, vmirror=0.0, **kwargs):
	p = {}
	if hmirror != None:
		p['horz_reflect_prob'] = hmirror
	
	if vmirror != None:
		p['vert_reflect_prob'] = vmirror

	return p
	

def createNoiseParam(low, high=None):
	std = [low]

	if high != None:
		std.append(high)

	return dict(std_dev=std)

def createNoiseParamE(params):
	return dict(std_dev=params)

def createRotateParam(rotation):
	return dict(max_angle=rotation)

def createRotateParamE(params):
	return dict(min_angle=params[0], max_angle=params[1], prob_negative=params[2])

def createShearParam(shear):
	return dict(max_shear_angle=shear)

def createShearParamE(params):
	return dict(min_shear_angle=params[0], max_shear_angle=params[1], prob_negative=params[2], prob_horizontal=params[3])

def createBlurParam(blur):
	return dict(max_sigma = blur)

def createBlurParamE(params):
	return dict(min_sigma=params[0], max_sigma=params[1])

def createUnsharpParam(params):
	if isinstance(params, dict):
		return params
	else:
		return dict(max_sigma=params)

def createPerspectiveParam(sig):
	return dict(max_sigma=sig)
	
def createPerspectiveParamE(params):
	return dict(values=params)

def createElasticDeformationParam(elastic_sigma, elastic_max_alpha):
	return dict(sigma=elastic_sigma, max_alpha=elastic_max_alpha)

def createElasticDeformationParamE(params):
	return dict(sigma=params[0], min_alpha=params[1], max_alpha=params[2])


def createTransformParam2(scale, shift,  seed, rotate=False, shear=False, perspective=False, 
		elastic=False, color_jitter=False, blur=False, noise=False, zero_border=0, gt=False):
	if gt:
		interp = proto.INTER_NEAREST
	else:
		interp = proto.INTER_LINEAR
	params = []
	if scale is not None and shift is not None:
		params.append({'linear_params': {'scale': scale, 'shift': shift}})
	elif scale is not None:
		params.append({'linear_params': {'scale': scale, 'shift': 0}})
	elif shift is not None:
		params.append({'linear_params': {'scale': 1.0, 'shift': shift}})
	if rotate:
		params.append({'rotate_params': {'max_angle': 5, 
										'interpolation': interp,
										'border_mode': proto.BORDER_REFLECT}})
	if shear:
		params.append({'shear_params': {'max_shear_angle': 5, 
										'interpolation': interp,
										'border_mode': proto.BORDER_REFLECT}})
	if perspective:
		params.append({'perspective_params': {'max_sigma': 0.00005, 
											'interpolation': interp,
											'border_mode': proto.BORDER_REFLECT}})
	if elastic:
		params.append({'elastic_deformation_params': {'sigma': 3.0, 
												'max_alpha': 5,
												'interpolation': interp,
												'border_mode': proto.BORDER_REFLECT}})
	if blur:
		params.append({'gauss_blur_params': {'max_sigma': 1.2}})
	if noise:
		params.append({'gauss_noise_params': {'std_dev': 10./255}})
	if color_jitter:
		params.append({'color_jitter_params': {'sigma': 10./255}})
	if zero_border:
		params.append({'zero_border_params': {'zero_len': zero_border}})
	return {'params': params, 'rng_seed': seed}
		


def createTransformParam(phase, seed=None, test_transforms = DEFAULT_TEST_TRANSFORMS, deploy=False, **kwargs):
	params = []

	if deploy:
		tt = test_transforms
		transforms = {}
		for t in tt:
			transforms[t] = []
		#if not kwargs.get('crop'):
		#	transforms[1] = ['none']


	# resize
	if (phase == caffe.TRAIN or deploy) and 'sizes' in kwargs:
		sizes = kwargs['sizes']
		params.append(dict(resize_params = {'size': [sizes[0], sizes[-1]]}))
		if deploy:
			all_ts = list()
			for size in sizes:
				transforms[size] = ["resize %d %d" % (size, size)]
				all_ts.append(transforms[size][0])
			transforms['all'] = all_ts

	#noise
	if (phase == caffe.TRAIN or deploy) and 'noise_std' in kwargs:
		noise = kwargs['noise_std']

		if not isinstance(noise, list):
			noise = [noise]

		params.append(dict(gauss_noise_params = createNoiseParam(*noise)))

		if deploy:
			for t in tt:
				transforms[t].extend(make_transforms.make_gaussnoise_transforms(noise[1], t))

	if (phase == caffe.TRAIN or deploy) and 'salt_max_flip' in kwargs:
		max_flip_prob = kwargs['salt_max_flip']
		params.append(dict(salt_pepper_params = dict(max_percent_pixels=max_flip_prob)))

	# color jitter
	if (phase == caffe.TRAIN or deploy) and 'color_std' in kwargs:
		sigma = kwargs['color_std']

		params.append(dict(color_jitter_params = createColorJitterParam(sigma)))

		if deploy:
			for t in tt:
				transforms[t].extend(make_transforms.make_color_jitter_transforms(sigma, t))

	if kwargs.get('hsv'):
		params.append(dict(hsv_params = dict()))

	#linear
	if 'scale' in kwargs or 'shift' in kwargs:
		params.append(dict(linear_params = createLinearParam(**kwargs)))

	if phase == caffe.TRAIN or deploy:
		#mirror
		if 'hmirror' in kwargs or 'vmirror' in kwargs:
			params.append(dict(reflect_params = createReflectParam(**kwargs)))
			if deploy:
				h = kwargs.get('hmirror', 0)
				v = kwargs.get('vmirror', 0)

				if 'shear' not in kwargs and 'crop' not in kwargs:
					for t in tt:
						transforms[t].extend(make_transforms.make_mirror_transforms(h,v))
						break


		#Perspective
		if 'perspective' in kwargs:
			params.append(dict(perspective_params = createPerspectiveParam(kwargs['perspective'])))
			
			if deploy:
				for t in tt:
					transforms[t].extend(make_transforms.make_perspective_transforms(kwargs['perspective'], t))

		#Elastic
		if 'elastic_sigma' in kwargs:
			params.append(dict(elastic_deformation_params = createElasticDeformationParam(kwargs['elastic_sigma'], kwargs['elastic_max_alpha'])))
			
			if deploy:
				for t in tt:
					transforms[t].extend(make_transforms.make_elastic_deformation_transforms(kwargs['elastic_sigma'], kwargs['elastic_max_alpha'], t))

		#rotate
		if 'rotation' in kwargs:
			params.append(dict(rotate_params = createRotateParam(kwargs['rotation'])))
			if deploy:
				for t in tt:
					transforms[t].extend(make_transforms.make_rotation_transforms(kwargs['rotation'], t))

		if 'shear' in kwargs:
			params.append(dict(shear_params = createShearParam(kwargs['shear']))) 
		
			if deploy and 'hmirror' not in kwargs and 'vmirror' not in kwargs and 'crop' not in kwargs:
				for t in tt:
					transforms[t].extend(make_transforms.make_shear_transforms(kwargs['shear'], t))


		#blur
		p = {}
		if 'blur' in kwargs:
			p['gauss_blur_params'] = createBlurParam(kwargs['blur'])
		
			if deploy:
				split = 2 if 'unsharp' in kwargs else 1
				for t in tt:
					transforms[t].extend(make_transforms.make_blur_transforms(kwargs['blur'], t/split))


		#unsharp
		if 'unsharp' in kwargs:
			p['unsharp_mask_params'] = createUnsharpParam(kwargs['unsharp'])

			if deploy:
				split = 2 if 'blur' in kwargs else 1
				for t in tt:
					transforms[t].extend(make_transforms.make_unsharp_transforms(kwargs['unsharp'], t))


		if len(p) > 0:
			params.append(p)

	if kwargs.get('crop_perc'):
		params.append(dict(crop_params = dict(size_perc=[kwargs.get('crop_perc'), 1.])))

	#crop
	if kwargs.get('crop'):
		params.append(dict(crop_params = createCropParam(phase)))
 
		if deploy and 'hmirror' not in kwargs and 'vmirror' not in kwargs and 'shear' not in kwargs:
			for t in tt:
				im_size = kwargs['im_size']
				crop_size = kwargs['crop']
				transforms[t].extend(make_transforms.make_crop_transforms(im_size, crop_size, int(round(np.sqrt(t)))))

	# For combined data augmentation. This is pretty messy
	if deploy:
		h = kwargs.get('hmirror', 0)
		v = kwargs.get('vmirror', 0)
		im_size = kwargs.get('im_size', None)
		angle = kwargs.get('shear', None)
		repeats = kwargs.get('shear_repeats', 1)
		if 'crop' in kwargs and 'shear' in kwargs and ('hmirror' in kwargs or 'vmirror' in kwargs):
			transforms['crop_shear_mirror'] = make_transforms.make_crop_shear_mirror_transforms(im_size, 227, h, v, angle, repeats)

		if 'crop' in kwargs and 'shear' in kwargs:
			transforms['crop_shear'] = make_transforms.make_crop_shear_transforms(im_size, 227, angle, repeats)
			transforms['crop'] = make_transforms.make_crop_transforms(im_size, 227, 3)
			transforms['shear'] = make_transforms.make_shear_transforms(angle, 10) 

		if 'crop' in kwargs and ('hmirror' in kwargs or 'vmirror' in kwargs):
			transforms['crop_mirror'] = make_transforms.make_crop_mirror_transforms(im_size, 227, h, v)
			transforms['crop'] = make_transforms.make_crop_transforms(im_size, 227, 3)
			transforms['mirror'] = make_transforms.make_mirror_transforms(h, v)

		if 'shear' in kwargs and ('hmirror' in kwargs or 'vmirror' in kwargs):
			transforms['shear_mirror'] = make_transforms.make_shear_mirror_transforms( h, v, angle, repeats)
			transforms['mirror'] = make_transforms.make_mirror_transforms(h, v)
			transforms['shear'] = make_transforms.make_shear_transforms(angle, 10) 


	p = dict(params=params)

	if seed != None:
		p['rng_seed'] = seed

	if deploy and "transforms_folder" in kwargs:
		for t, trans in transforms.items():
			if len(trans) == 0:
				continue

			filename = os.path.join(kwargs["transforms_folder"], "transforms_%s.txt" % (t))
			#print trans
			with open(filename, "w") as f:
				f.write('\n'.join(trans))

	return p


def createBinarizeNetwork(train_input_sources=[], train_label_sources=[], train_label_equal_sources=[], val_input_sources=[], 
						  val_label_sources=[], val_label_equal_sources=[], uniform_weights=False, depth=3, kernel_size=3, 
						  num_filters=24, num_scales=1, lrn=0, pool=0, wfm_loss=True, global_features=0, one_convs=0, deploy=False, 
						  seed=None, rotate=False, shear=False, perspective=False, elastic=False, color_jitter=False, blur=False, 
						  noise=False, zero_border=0, train_batch_size=1, pfm_loss_weight=0, densenet=False, recall_shift=0,
						  residual=False, margin=0.5):
	assert deploy or len(train_input_sources) == len(val_input_sources)
	assert deploy or len(train_label_sources) == len(val_label_sources)
	assert deploy or train_input_sources
	if seed == None:
		seed = random.randint(0, 2147483647)
	n = caffe.NetSpec()	
	data_param = dict(backend=P.Data.LMDB)

	if deploy:
		n.data = L.Input()
	else:
		# training inputs
		inputs = list()

		for source in train_input_sources:
			input = L.DocData(sources=[source], include=dict(phase=caffe.TRAIN), batch_size=train_batch_size, backend=P.Data.LMDB,
				seed=seed, rand_advance_skip=10,
				image_transform_param=createTransformParam2(scale=(1./255), shift=127, seed=seed, 
					rotate=rotate, shear=shear, perspective=perspective, elastic=elastic, color_jitter=color_jitter,
					blur=blur, noise=noise))
			inputs.append(input)

		if len(inputs) == 1:
			n.data = inputs[0]
		else:
			n.data = L.Concat(*inputs, include=dict(phase=caffe.TRAIN))

		# training labels
		n.gt = L.DocData(sources=[train_label_sources[0]], include=dict(phase=caffe.TRAIN), batch_size=train_batch_size, backend=P.Data.LMDB,
			seed=seed, rand_advance_skip=10,
			image_transform_param=createTransformParam2(scale=1, shift=None, seed=seed, gt=True,
				rotate=rotate, shear=shear, perspective=perspective, elastic=elastic, color_jitter=False, blur=False, noise=False))

		# good value for recall_shift is -64 to encourage border pixels to be ink
		n.recall_weights = L.DocData(sources=[train_label_sources[1]], include=dict(phase=caffe.TRAIN), batch_size=train_batch_size, backend=P.Data.LMDB,
			seed=seed, rand_advance_skip=10,
			image_transform_param=createTransformParam2(scale=(1./128), shift=(-128. * recall_shift), seed=seed, gt=True,
				rotate=rotate, shear=shear, perspective=perspective, elastic=elastic, color_jitter=False, blur=False, noise=False, zero_border=zero_border))
		n.precision_weights = L.DocData(sources=[train_label_sources[2]], include=dict(phase=caffe.TRAIN), batch_size=train_batch_size, backend=P.Data.LMDB,
			seed=seed, rand_advance_skip=10,
			image_transform_param=createTransformParam2(scale=(1./128), shift=None, seed=seed, gt=True,
				rotate=rotate, shear=shear, perspective=perspective, elastic=elastic, color_jitter=False, blur=False, noise=False, zero_border=zero_border))

		if wfm_loss and uniform_weights:
			n.recall_equal_weights = L.DocData(sources=[train_label_equal_sources[1]], include=dict(phase=caffe.TRAIN), batch_size=train_batch_size, backend=P.Data.LMDB,
				seed=seed, rand_advance_skip=10,
				image_transform_param=createTransformParam2(scale=(1./128), shift=None, seed=seed, gt=True,
					rotate=rotate, shear=shear, perspective=perspective, elastic=elastic, color_jitter=False, blur=False, noise=False, zero_border=zero_border))
			n.precision_equal_weights = L.DocData(sources=[train_label_equal_sources[2]], include=dict(phase=caffe.TRAIN), batch_size=train_batch_size, backend=P.Data.LMDB,
				seed=seed, rand_advance_skip=10,
				image_transform_param=createTransformParam2(scale=(1./128), shift=None, seed=seed, gt=True,
					rotate=rotate, shear=shear, perspective=perspective, elastic=elastic, color_jitter=False, blur=False, noise=False, zero_border=zero_border))
			
		# val inputs
		inputs = list()

		for source in val_input_sources:
			input = L.DocData(sources=[source], include=dict(phase=caffe.TEST), batch_size=1, backend=P.Data.LMDB,
				image_transform_param=createTransformParam2(scale=(1./255), shift=127, seed=seed))
			inputs.append(input)

		if len(inputs) == 1:
			n.VAL_data = inputs[0]
		else:
			n.VAL_data = L.Concat(*inputs, include=dict(phase=caffe.TEST))
			
		# val labels
		n.VAL_gt = L.DocData(sources=[val_label_sources[0]], include=dict(phase=caffe.TEST), batch_size=1, backend=P.Data.LMDB,
			image_transform_param=createTransformParam2(scale=1, shift=None, seed=seed, gt=True))

		n.VAL_recall_weights = L.DocData(sources=[val_label_sources[1]], include=dict(phase=caffe.TEST), batch_size=1, backend=P.Data.LMDB,
			image_transform_param=createTransformParam2(scale=(1./128), shift=(-128 * recall_shift), seed=seed, gt=True))
		n.VAL_precision_weights = L.DocData(sources=[val_label_sources[2]], include=dict(phase=caffe.TEST), batch_size=1, backend=P.Data.LMDB,
			image_transform_param=createTransformParam2(scale=(1./128), shift=None, seed=seed, gt=True))

		if wfm_loss and uniform_weights:
			n.VAL_recall_equal_weights = L.DocData(sources=[val_label_equal_sources[1]], include=dict(phase=caffe.TEST), batch_size=1, backend=P.Data.LMDB,
				image_transform_param=createTransformParam2(scale=(1./128), shift=None, seed=seed, gt=True))
			n.VAL_precision_equal_weights = L.DocData(sources=[val_label_equal_sources[2]], include=dict(phase=caffe.TEST), batch_size=1, backend=P.Data.LMDB,
				image_transform_param=createTransformParam2(scale=(1./128), shift=None, seed=seed, gt=True))

	# middle layers
	prev_layer = n.data
	pad_size = (kernel_size - 1) / 2
	layers = collections.defaultdict(list)
	layers[0].append(prev_layer)

	# the network can operate over multiple scales of the image (e.g. 1, 1/2, 1/4) and each
	# scale is like its own CNN. In the end, the last layer in each scale gets upsampled to
	# the original size and concatenated
	for scale in xrange(num_scales):
		for conv_idx in xrange(depth - scale): 
			do_lrn = lrn > (conv_idx + scale)

			# do the densenet thing
			if densenet and conv_idx > 0:
				prev_layer = L.Concat(*layers[scale])

			if residual and conv_idx > 0:
				residual_layer = convLayer(prev_layer, kernel_size=kernel_size, pad=pad_size, num_output=num_filters, stride=1, lrn=do_lrn)
				prev_layer = L.Eltwise(residual_layer, prev_layer)
			else:
				# create a single conv layer
				prev_layer = convLayer(prev_layer, kernel_size=kernel_size, pad=pad_size, num_output=num_filters, stride=1, lrn=do_lrn)

			# This does max-pooling without downsampling
			if pool > (conv_idx + scale):
				prev_layer = poolLayer(prev_layer, kernel_size=3, stride=1, pad=1)
			layers[scale].append(prev_layer)

		# if not last scale
		if scale < (num_scales - 1):
			# do downsampling for the next scale
			#prev_layer = poolLayer(layers[scale][0], kernel_size=2, stride=2)

			# first scale should pool the result of the first convolution, not the data
			# layers[scale >=1][0] is the result of a convolution on the pooled data
			# so the start of each scale should be the result of [conv, (pool, conv)*]
			idx = 0 if scale else 1
			prev_layer = L.Pooling(layers[scale][idx], pool=P.Pooling.AVE, kernel_size=2, stride=2)

	# collapse into a 1x1 scale
	if global_features > 0:
		prev_layer = convLayer(layers[num_scales-1][0], kernel_size=kernel_size, pad=pad_size, num_output=num_filters, stride=2)
		prev_layer = L.Pooling(prev_layer, pool=P.Pooling.AVE, global_pooling=True)
		for conv_idx in xrange(global_features): 
			prev_layer = convLayer(prev_layer, kernel_size=1, pad=0, num_output=num_filters, stride=1)
			layers[num_scales].append(prev_layer)

	# collect the last layers in each scale
	last_layers = []
	for scale in xrange(num_scales + (1 if global_features > 0 else 0)):
		scale_layers = layers[scale]
		if scale_layers:
			last_layers.append(scale_layers[-1])
	
	# resize smaller scales to original size
	if len(last_layers) > 1:
		for idx in xrange(len(last_layers)):
			if idx == 0:
				continue
			last_layers[idx] = L.BilinearInterpolation(last_layers[idx], n.data)
		n.merged = L.Concat(*last_layers)
		prev_layer = convLayer(n.merged, kernel_size=kernel_size, pad=pad_size, num_output=num_filters, stride=1)
		layers[0].append(prev_layer)

	# apply any number of 1x1 convolutions
	for idx in xrange(one_convs):
		prev_layer = convLayer(prev_layer, kernel_size=1, pad=0, num_output=num_filters, stride=1)
		layers[0].append(prev_layer)

	if densenet:
		prev_layer = L.Concat(*layers[0])
		
	# output/loss layer
	if deploy:
		n.prob = convLayerSigmoid(prev_layer, kernel_size=kernel_size, pad=pad_size, num_output=1, stride=1)
	elif wfm_loss:
		prev_layer = convLayerSigmoid(prev_layer, kernel_size=kernel_size, pad=pad_size, num_output=1, stride=1)
		if uniform_weights:
			n.equal_fmeasure_loss = L.WeightedFmeasureLoss(prev_layer, n.gt, n.recall_equal_weights, n.precision_equal_weights)

			# for metric purposes only
			n.weighted_fmeasure_loss = L.WeightedFmeasureLoss(prev_layer, n.gt, n.recall_weights, n.precision_weights, 
				loss_weight=pfm_loss_weight, margin=margin)
		else:
			n.weighted_fmeasure_loss = L.WeightedFmeasureLoss(prev_layer, n.gt, n.recall_weights, n.precision_weights, margin=margin)

		# for metric purposes
		n.precision, n.recall, n.accuracy, n.nmr, n.fmeasure = L.PR(prev_layer, n.gt, ntop=5, include=dict(phase=caffe.TEST))
	else:
		prev_layer = convLayerOnly(prev_layer, kernel_size=kernel_size, pad=pad_size, num_output=1, stride=1)
		n.class_loss = L.SigmoidCrossEntropyLoss(prev_layer, n.gt)
		prev_layer = L.Sigmoid(prev_layer)

		# for metric purposes only
		n.precision, n.recall, n.accuracy, n.nmr, n.fmeasure = L.PR(prev_layer, n.gt, ntop=5, include=dict(phase=caffe.TEST))
		n.weighted_fmeasure_loss = L.WeightedFmeasureLoss(prev_layer, n.gt, n.VAL_recall_weights, n.VAL_precision_weights, 
			loss_weight=pfm_loss_weight, margin=margin)

	return n.to_proto()


def createTransformParamE(seed=None, val=False, **kwargs):
	params = []

	# gauss noise
	if 'noise' in kwargs:
		params.append(dict(gauss_noise_params=createNoiseParamE(kwargs['noise'])))

	# color jitter
	if 'color' in kwargs:
		params.append(dict(color_jitter_params=createColorJitterParamE(kwargs['color'])))

	# linear
	if 'scale' in kwargs or 'shift' in kwargs:
		params.append(dict(linear_params = createLinearParam(**kwargs)))

	# mirrors
	if 'hmirror' in kwargs or 'vmirror' in kwargs:
		params.append(dict(reflect_params=createReflectParam(**kwargs)))

	# perspective
	if 'perspective' in kwargs:
		params.append(dict(perspective_params=createPerspectiveParamE(kwargs['perspective'])))

	# elastic
	if 'elastic' in kwargs:
		params.append(dict(elastic_deformation_params=createElasticDeformationParamE(kwargs['elastic'])))

	# rotate
	if 'rotation' in kwargs:
		if val:
			kwargs = dict(**kwargs)
			kwargs['rotation'][0] = 0.5 * (kwargs['rotation'][0] + kwargs['rotation'][1])
			kwargs['rotation'][1] = kwargs['rotation'][0] + 0.01;
		params.append(dict(rotate_params=createRotateParamE(kwargs['rotation'])))

	# shear
	if 'shear' in kwargs:
		if val:
			kwargs = dict(**kwargs)
			kwargs['shear'][0] = 0.5 * (kwargs['shear'][0] + kwargs['shear'][1])
			kwargs['shear'][1] = kwargs['shear'][0] + 0.01;
		params.append(dict(shear_params=createShearParamE(kwargs['shear']))) 
	
	# blur
	if 'blur' in kwargs:
		params.append(dict(gauss_blur_params=createBlurParamE(kwargs['blur'])))

	# unsharp
	if 'unsharp' in kwargs:
		params.append(dict(unsharp_mask_params=createUnsharpParamE(kwargs['unsharp'])))

	# crop
	if 'crop' in kwargs:
		params.append(dict(crop_params=createCropParamE(kwargs['crop'])))
	
	p = dict(params=params)
	if seed != None:
		p['rng_seed'] = seed
	return p
	

def createEquivarianceNetwork(sources=[], val_sources=[], num_output=1, batch_size=8, deploy=False, seed_t=None, shift_channels=None,
					scale_channels=None, val_batch_size=VAL_BATCH_SIZE, l_tparams=[{}], mapping='identity', total_l2_loss_weight=2.,
					total_non_first_ce_loss_weight=1, num_hidden=1000, seed_d=None):
	n = caffe.NetSpec()	

	if seed_t is None:
		seed_t = random.randint(0, 2147483647)
	if seed_d is None:
		seed_d = random.randint(0, 2147483647)

	data_param = dict(backend=P.Data.LMDB, seed=seed_d)

	# divide the total loss among the non-default individual transforms
	l2_loss_weight = total_l2_loss_weight / (len(l_tparams) - 1) if len(l_tparams) > 1 else 0
	non_first_ce_loss_weight = total_non_first_ce_loss_weight / (len(l_tparams) - 1) if len(l_tparams) > 1 else 0
	
	for t_idx, tparams in enumerate(l_tparams):
		
		# handle inputs
		if deploy:
			data = L.Input()
			cur_layer = data

		elif len(sources) == 1:
			data, labels = L.DocData(sources = [sources[0]], include=dict(phase=caffe.TRAIN), batch_size=batch_size, 
					image_transform_param=createTransformParamE(seed=seed_t, shift=shift_channels[0], scale=scale_channels[0], **tparams), 
					label_names=["dbid"], ntop=2, rand_advance_skip=3,**data_param)
			setattr(n, "data_%d" % t_idx, data)
			setattr(n, "labels_%d" % t_idx, labels)

			if val_sources:
				val_data, val_labels = L.DocData(sources = [val_sources[0]], include=dict(phase=caffe.TEST), batch_size=val_batch_size,
						image_transform_param=createTransformParamE(shift=shift_channels[0], scale=scale_channels[0], val=True, **tparams), 
						label_names=["dbid"], ntop=2, **data_param)
				setattr(n, "VAL_data_%d" % t_idx, val_data)
				setattr(n, "VAL_labels_%d" % t_idx, val_labels)

		else:
			first, labels = L.DocData(sources = [sources[0]], include=dict(phase=caffe.TRAIN), batch_size=batch_size, 
					image_transform_param=createTransformParamE(seed=seed_t, shift=shift_channels[0], scale=scale_channels[0], **tparams), 
					label_names=["dbid"], ntop=2, rand_advance_skip=3, **data_param)

			inputs = map(lambda s, t: L.DocData(sources=[s], include=dict(phase=caffe.TRAIN), batch_size=batch_size, rand_advance_skip=3,
				image_transform_param=createTransformParamE(seed=seed_t, shift=t[0], scale=t[1], **tparams),
				**data_param), 
				sources[1:], 
				zip(shift_channels[1:], scale_channels[1:]))
		
			#print inputs
			data = L.Concat(first, *inputs, include=dict(phase=caffe.TRAIN))
			setattr(n, "data_%d" % t_idx, data)
			setattr(n, "labels_%d" % t_idx, labels)

			if val_sources:
				val_first, val_labels = L.DocData(sources = [val_sources[0]], include=dict(phase=caffe.TEST), batch_size=val_batch_size,
						image_transform_param=createTransformParamE(seed=seed_t, shift=shift_channels[0], scale=scale_channels[0], val=True, **tparams), 
						label_names=["dbid"], ntop=2, **data_param)

				val_inputs = map(lambda s, t: L.DocData(sources=[s], include=dict(phase=caffe.TEST), batch_size=val_batch_size, 
					image_transform_param=createTransformParamE(seed=seed_t, shift=t[0], scale=t[1], **tparams), 
					**data_param), 
					val_sources[1:], zip(shift_channels[1:],scale_channels[1:]))
		
				val_data = L.Concat(val_first, *val_inputs, name="val_data-%d" % t_idx, include=dict(phase=caffe.TEST))
				setattr(n, "VAL_data_%d" % t_idx, val_data)
				setattr(n, "VAL_labels_%d" % t_idx, val_labels)

		cur_layer = data
		layers = BEFORE_LAYERS
		for layer_func, kwargs in layers:
			kwargs = kwargs.copy()
			kwargs['name'] = kwargs['name'] + "-%d" % t_idx
			cur_layer = layer_func(cur_layer, **kwargs)
			

		if t_idx:
			l2_loss = L.EuclideanLoss(cur_layer, layer_to_predict, loss_weight=l2_loss_weight/2, 
				name="invar_map_%d_loss" % t_idx, loss_param=dict(normalize=True)) 
			setattr(n, "l2_in_loss_%d" % t_idx, l2_loss)

			if mapping == 'identity':
				pass
			elif mapping == 'linear':
				linear_layer = ipLayer(cur_layer, name="linear_mapping-%d" % t_idx, num_output=4096)
				cur_layer = L.Eltwise(linear_layer, cur_layer)  # residual layer
			elif mapping == 'mlp':
				mlp_layer = ipLayer(cur_layer, name="hidden_mapping-%d" % t_idx, num_output=num_hidden)
				mlp_layer = L.ReLU(mlp_layer, in_place=True)
				mlp_layer = ipLayer(mlp_layer, name="linear_mapping-%d" % t_idx, num_output=4096)
				cur_layer = L.Eltwise(mlp_layer, cur_layer)  # residual layer

			cur_layer = L.ReLU(cur_layer, in_place=True)
			l2_loss = L.EuclideanLoss(cur_layer, layer_to_predict, loss_weight=l2_loss_weight, 
				name="%s_map_%d_loss" % (mapping, t_idx), loss_param=dict(normalize=True)) 
			setattr(n, "l2_eq_loss_%d" % t_idx, l2_loss)
		else:
			# this is the transform (or lack thereof) each other is trying to predict
			layer_to_predict = L.ReLU(cur_layer, in_place=False)
			cur_layer = layer_to_predict

		layers = AFTER_LAYERS
		for layer_func, kwargs in layers:
			kwargs = kwargs.copy()
			kwargs['name'] = kwargs['name'] + "-%d" % t_idx
			cur_layer = layer_func(cur_layer, **kwargs)

		top = ipLayer(cur_layer, name="top-%d" % t_idx, num_output=num_output, param_name='fc8')
		if deploy:
			n.prob = L.Softmax(top, name='prob')
		else:
			# the first "transform" is typically the identity, so we might want to assign all other loss weights
			# to be somewhat lower, so the training focuses more on classifying untransformed images with a bias
			# toward also classifying the transformed images (after mapping them back into an untransformed 
			# representation)
			loss_weight = non_first_ce_loss_weight if t_idx else 1.   
			loss = L.SoftmaxWithLoss(top, labels, name="ce_loss-%d" % t_idx, loss_weight=loss_weight)
			accuracy = L.Accuracy(top, labels, name="accuracy-%d" % t_idx)
			setattr(n, "loss_%d" % t_idx, loss)
			setattr(n, "accuracy_%d" % t_idx, accuracy)

	return n.to_proto()



def createNetwork(sources, size, val_sources=None,  num_output=1000, concat=False, pool=None, batch_size=32, deploy=False, 
					seed=None, shift_channels=None, scale_channels=None, multiple=False, val_batch_size=VAL_BATCH_SIZE, bn=False,
					dropout=True, lrn=True, **tparams):
	n = caffe.NetSpec()	
	#data
	data_param = dict(backend=P.Data.LMDB)

	if len(sources) == 1:
		concat = False
	
	#Helper function for checking transform params
	def checkTransform(trans, default):
		#If trans is not defined, replace with default
		if not trans:
			trans = default

		#If Shift channels is only one value, 
		if (not isinstance(trans, list)):
			trans = [trans]*len(sources)

		return trans

	shift_channels = checkTransform(shift_channels, 0)
	scale_channels = checkTransform(scale_channels, 1.0)

	if seed == None:
		seed = random.randint(0, 2147483647)
	
	if not deploy:
		if concat:
			first, targets = L.DocData(sources = [sources[0]], include=dict(phase=caffe.TRAIN), batch_size=batch_size, 
					image_transform_param=createTransformParam(caffe.TRAIN, seed=seed, shift=shift_channels[0], scale=scale_channels[0], **tparams), 
					label_names=["dbid"], ntop=2, **data_param)

		
			if val_sources:
				val_first, val_targets = L.DocData(sources = [val_sources[0]], include=dict(phase=caffe.TEST), batch_size=val_batch_size,
						image_transform_param=createTransformParam(caffe.TEST, shift=shift_channels[0], scale=scale_channels[0], **tparams), 
						label_names=["dbid"], ntop=2, **data_param)

				tparams['hsv'] = False

				val_inputs = map(lambda s, t: L.DocData(sources=[s], include=dict(phase=caffe.TEST), batch_size=val_batch_size, 
					image_transform_param=createTransformParam(caffe.TEST, shift=t[0], scale=t[1], **tparams), **data_param), 
					val_sources[1:], zip(shift_channels[1:],scale_channels[1:]))
		
				n.VAL_data = L.Concat(val_first, *val_inputs, name="val_data", include=dict(phase=caffe.TEST))
				n.VAL_labels = val_targets


			#print len(sources)
			inputs = map(lambda s, t: L.DocData(sources=[s], include=dict(phase=caffe.TRAIN), batch_size=batch_size, 
				image_transform_param=createTransformParam(caffe.TRAIN, seed=seed, shift=t[0], scale=t[1], **tparams), **data_param), sources[1:], 
				zip(shift_channels[1:], scale_channels[1:]))
		
			#print inputs
			n.data = L.Concat(first, *inputs, include=dict(phase=caffe.TRAIN))
			n.labels = targets
			
	
		else:
			data_param['ntop'] = 2
			data_param['label_names'] = ["dbid"]

			if multiple:
				# enable weight by size because we have multiple lmdbs to read from
				n.data, n.labels = L.DocData(sources = sources, batch_size=batch_size, include=dict(phase=caffe.TRAIN), 
					image_transform_param=createTransformParam(caffe.TRAIN, seed=seed, shift=shift_channels[0], scale=scale_channels[0], **tparams), 
					weights_by_size=True, **data_param)
			else:
				n.data, n.labels = L.DocData(sources = sources, batch_size=batch_size, include=dict(phase=caffe.TRAIN), 
					image_transform_param=createTransformParam(caffe.TRAIN, seed=seed, shift=shift_channels[0], scale=scale_channels[0], **tparams), 
					**data_param)


			if val_sources:
				if multiple:
					# make sure in_order and no_wrap are true so we can iterate the whole validation set through multiple lmdbs
					n.VAL_data, n.VAL_labels = L.DocData(sources=val_sources, name="validation", batch_size=val_batch_size, 
						include=dict(phase=caffe.TEST),  
						image_transform_param=createTransformParam(caffe.TEST, shift=shift_channels[0], scale=scale_channels[0], **tparams), 
						in_order=True, no_wrap=True, **data_param) 
				else:
					n.VAL_data, n.VAL_labels = L.DocData(sources=val_sources, name="validation", batch_size=val_batch_size, 
						include=dict(phase=caffe.TEST), 
						image_transform_param=createTransformParam(caffe.TEST, shift=shift_channels[0], scale=scale_channels[0], **tparams), 
						**data_param) 
	else:
		createTransformParam(caffe.TEST, shift=shift_channels[0], scale=scale_channels[0], deploy=deploy, **tparams)
		n.data = L.Input()

	#CONV layers
	if 'depth' in tparams:
		layers = DEPTH_CONV_LAYERS[tparams['depth']]
	else:
		layers = CONV_LAYERS[size]
	layer = n.data
	for t, kwargs in layers[:-1]:
		if (tparams.get('width_mult') or tparams.get('conv_width_mult')) and kwargs.get('num_output'):
			kwargs = kwargs.copy()
			mult = tparams.get('width_mult')
			if not mult:
				mult = tparams.get('conv_width_mult')
			kwargs['num_output'] = int(mult * kwargs['num_output'])
		if kwargs.get('num_output'):
			kwargs['bn'] = bn
		if 'norm' in kwargs['name'] and not lrn:
			continue
		layer = t(layer, **kwargs)

	if pool is not None:
		# add in a padding to powers of 2 so that we guarentee the same size output
		layer = L.Padding(layer, pad_to_power_of_2=True, name="padding")
		if pool == 'spp':
			layer = L.SPP(layer, pyramid_height=4, name="spp")
		elif pool == 'hvp':
			layer = L.HVP(layer, num_horz_partitions=3, num_vert_partitions=3, name="hvp")
	else:
		layer = layers[-1][0](layer, **layers[-1][1])

			
	
	#FC layers
	if 'fc_depth' in tparams:
		fc_layers = DEPTH_FC_LAYERS[tparams['fc_depth']]
	else:
		fc_layers = FC_LAYERS[size]
	for t, kwargs in fc_layers:
		if (tparams.get('width_mult') or tparams.get('fc_width_mult')) and kwargs.get('num_output'):
			kwargs = kwargs.copy()
			mult = tparams.get('width_mult')
			if not mult:
				mult = tparams.get('fc_width_mult')
			kwargs['num_output'] = int(mult * kwargs['num_output'])
		if kwargs.get('num_output'):
			kwargs['bn'] = bn
		if 'dropout' in kwargs['name'] and not dropout:
			continue
		layer = t(layer, **kwargs)

	#Output Layer
	top = ipLayer(layer, name="top", num_output=num_output)

	n.top = top

	if not deploy:
		n.loss = L.SoftmaxWithLoss(n.top, n.labels)

		n.accuracy = L.Accuracy(n.top, n.labels)
	else:
		n.prob = L.Softmax(n.top)
	
	return n.to_proto()


def createBinarizeExperiment(ds, tags, group, experiment, num_experiments=1, wfm_loss=True, lr=0.001, uniform_weights=False,
		input_size=256, baseline_width=1, recall_weights="default", **kwargs):

	sources_input_train = map(lambda tag: LMDB_PATH_BINARIZE(ds, tag, input_size, data_partition='train'), tags)
	sources_input_val = map(lambda tag: LMDB_PATH_BINARIZE(ds, tag, input_size, data_partition='val'), tags)
	sources_input_test = map(lambda tag: LMDB_PATH_BINARIZE(ds, tag, input_size, data_partition='test'), tags)

	if recall_weights == 'dilated':
		recall_tag = 'dilated_recall_weights'
	elif recall_weights == 'modified':
		recall_tag = 'modified_recall_weights'
	else:
		recall_tag = 'recall_weights'

	# weights for Pseudo F-measure
	#label_tags = ['baselines_%d' % baseline_width, recall_tag, 'precision_weights']
	label_tags = ['baselines_%d' % baseline_width, 'uniform_recall_weights', 'uniform_precision_weights']
	sources_label_train = map(lambda tag: LMDB_PATH_BINARIZE(ds, tag, input_size, data_partition='train'), label_tags)
	sources_label_val = map(lambda tag: LMDB_PATH_BINARIZE(ds, tag, input_size, data_partition='val'), label_tags)
	sources_label_test = map(lambda tag: LMDB_PATH_BINARIZE(ds, tag, input_size, data_partition='test'), label_tags)

	# weights for Equal weights F-measure
	label_tags = ['baselines_%d' % baseline_width, 'uniform_recall_weights', 'uniform_precision_weights']
	sources_label_equal_train = map(lambda tag: LMDB_PATH_BINARIZE(ds, tag, input_size, data_partition='train'), label_tags)
	sources_label_equal_val = map(lambda tag: LMDB_PATH_BINARIZE(ds, tag, input_size, data_partition='val'), label_tags)
	sources_label_equal_test = map(lambda tag: LMDB_PATH_BINARIZE(ds, tag, input_size, data_partition='test'), label_tags)

	params = {'train_input_sources': sources_input_train, 'train_label_sources': sources_label_train, 
			  'train_label_equal_sources': sources_label_equal_train, 'wfm_loss': wfm_loss, 'uniform_weights': uniform_weights}
	params.update(kwargs)
	
	for exp_num in range(1,num_experiments+1):
		exp_num = str(exp_num)

		out_dir = OUTPUT_FOLDER_BINARIZE(ds, group, experiment, exp_num)
		print out_dir

		if not os.path.exists(out_dir):
			print "Directory Not Found, Creating"
			os.makedirs(out_dir)

		#create train_train file
		train_val = os.path.join(out_dir, TRAIN_TRAIN)
		with open(train_val, "w") as f:
			n = str(createBinarizeNetwork(val_input_sources=sources_input_train, val_label_sources=sources_label_train, 
										  val_label_equal_sources=sources_label_equal_train, **params))
			f.write(re.sub("VAL_", "", n))

		#create train_val file
		train_val = os.path.join(out_dir, TRAIN_VAL)
		with open(train_val, "w") as f:
			n = str(createBinarizeNetwork(val_input_sources=sources_input_val, val_label_sources=sources_label_val, 
										  val_label_equal_sources=sources_label_equal_val, **params))
			f.write(re.sub("VAL_", "", n))
	
		#Create train_test file
		train_test = os.path.join(out_dir, TRAIN_TEST)
		with open(train_test, "w") as f:
			n = str(createBinarizeNetwork(val_input_sources=sources_input_test, val_label_sources=sources_label_test, 
										  val_label_equal_sources=sources_label_equal_test, **params))
			f.write(re.sub("VAL_", "", n))

		#Create Deploy File
		deploy_file = os.path.join(out_dir, DEPLOY_FILE)
		with open(deploy_file, "w") as f:
			n = createBinarizeNetwork(deploy=True, **params)
			for i, l in enumerate(n.layer):
				if l.type == "Input":
					del n.layer[i]
					break

			n.input.extend(['data'])
			n.input_dim.extend([1, get_num_channels(sources_input_train),256,256])
			f.write(str(n))

		input_file = os.path.join(out_dir, "inputs.txt")
		with open(input_file, "w") as f:
			for tag in tags:
				if isinstance(tag, tuple):
					tag = tag[1]
				for idx in xrange(len(tag) - 1):
					if tag[idx] == '_' and tag[idx +1].isdigit():
						tag = tag[:idx] + '/' + tag[idx+1:]
				f.write("%s\n" % tag)

		exp_folder = EXPERIMENTS_FOLDER_BINARIZE(ds,group,experiment,exp_num)
		snapshot_solver = os.path.join(exp_folder, SNAPSHOT_FOLDER, experiment)
		train_val_solver = os.path.join(exp_folder, TRAIN_VAL)

		solver = os.path.join(out_dir, SOLVER)
		with open(solver, "w") as f:
			f.write("net: \"%s\"\n" % (train_val_solver))
			f.write("base_lr: %f\n" % lr)
			f.write("gamma: %f\n" % 0.1)
			f.write("monitor_test: true\n")
			f.write("monitor_test_id: 0\n")
			f.write("test_compute_loss: true\n")
			f.write("max_steps_without_improvement: %d\n" % 4)
			f.write("max_periods_without_improvement: %d\n" % 3)
			f.write("min_iters_per_period: %d\n" % 2000)
			f.write("min_iters_per_period: %d\n" % 2000)
			f.write("min_iters_per_period: %d\n" % 2000)
			f.write("min_lr: %f\n" % 1e-6)
			if ds.startswith('synthetic'):
				f.write("max_iter: %d\n" % 800000)
			else:
				f.write("max_iter: %d\n" % 200000)
			f.write("max_nonfinite_test_loss: %d\n" % 1)
			f.write("clip_gradients: %f\n" % 10.)

			f.write("test_iter: %d\n" % lmdb_num_entries(sources_label_val[0]))
			batch_size = kwargs.get('train_batch_size', 4)
			f.write("test_interval: %d\n" % (lmdb_num_entries(sources_label_train[0]) / (2 * batch_size)))
			f.write("snapshot: %d\n" % (lmdb_num_entries(sources_label_train[0]) / (2 * batch_size)))
			f.write("momentum: %f\n" % 0.9)
			f.write("weight_decay: %f\n" % 0.0005)

			f.write("display: %d\n" % 100)
			f.write("solver_mode: GPU\n")
			f.write("snapshot_prefix: \"%s\"" % (snapshot_solver))
	

def createExperiment(ds, tags, group, experiment, num_experiments=1, pool=None, multiple=False, dsurf=False, 
		shift=None, scale=None, bn=False, dropout=True, lrn=True, **tparams):

	# Check if tags are all the same size or not
	# If they aren't we are doing multi-scale training, and need to stick them all
	# in the same doc data layer 
	if not isinstance(tags, list):
		tags = [tags]
	sizes = map(getSizeFromTag, tags)
	size = sizes[0]
	same_size = (not multiple)  # multiple AR training defaults to not the same size
	for s in sizes:
		same_size = (same_size and s == size)

	im_size = size
	tags_noSize = map(getTagWithoutSize, tags)
	if shift == "mean":
		shift = []
		for idx, t in enumerate(tags_noSize):
			if 'dsurf' in t:
				shift += 22 * [MEAN_VALUES[ds][t]]
			elif idx == 0 and tparams.get('hsv'):
				shift += [MEAN_VALUES[ds]['hsv']]
			else:
				shift += [MEAN_VALUES[ds][t]]

	if tparams.get('crop'):
		same_size = True
		size = tparams['crop']
	
	#if sizes are different, spatial pyramid pooling is required.
	if not same_size and pool is None:
		raise Exception("Input DBs are not the same size and regular pooling is enabled")

	for exp_num in range(1,num_experiments+1):
		exp_num = str(exp_num)

		out_dir = OUTPUT_FOLDER(ds, group, experiment, exp_num)
		print out_dir

		if not os.path.exists(out_dir):
			print "Directory Not Found, Creating"
			os.makedirs(out_dir)
		tf = TRANSFORMS_FOLDER(ds, group, experiment, exp_num)
		if not os.path.exists(tf):
			os.makedirs(tf)
		
		if multiple or dsurf:
			sources_tr, sources_val, sources_ts = [], [], []
			for tag in tags:
				if 'multiple' in tag or 'dsurf' in tag:
					tr, val, ts = LMDB_MULTIPLE_PATH(ds, tag, "1")
				else:
					# if we just have str, then the += lines will concatenate every character
					tr, val, ts = tuple(map(lambda s: [s], LMDB_PATH(ds, tag, "1")))
				sources_tr += tr
				sources_val += val
				sources_ts += ts
		else:
			# only 1 lmdb split is in current use
			sources = map(lambda t: LMDB_PATH(ds, t, "1"), tags)
			sources_tr, sources_val, sources_ts =  zip(*sources)

		#common parameters
		params = dict(sources=list(sources_tr), size=size, num_output=OUTPUT_SIZES[ds], concat=same_size, 
					pool=pool, shift_channels=shift, scale_channels=scale, batch_size=BATCH_SIZE[ds], 
					multiple=multiple, dsurf=dsurf, bn=bn, dropout=dropout, lrn=lrn, **tparams)
		if ds == "imagenet":
			params['val_batch_size'] = 50
		elif ds == "combined":
			params['val_batch_size'] = 80
		else:
			params['val_batch_size'] = VAL_BATCH_SIZE 

		#create train_val file
		train_val = os.path.join(out_dir, TRAIN_VAL)
		with open(train_val, "w") as f:
			n = str(createNetwork(val_sources=list(sources_val), **params))
			f.write(re.sub("VAL_", "", n))
	
		#Create train_test file
		train_test = os.path.join(out_dir, TRAIN_TEST)
		with open(train_test, "w") as f:
			n = str(createNetwork(val_sources=list(sources_ts), **params))
			f.write(re.sub("VAL_", "", n))

		#Create Deploy File
		deploy_file = os.path.join(out_dir, DEPLOY_FILE)
		with open(deploy_file, "w") as f:
			n = createNetwork(deploy=True, transforms_folder=tf,im_size=im_size, **params)
			for i, l in enumerate(n.layer):
				if l.type == "Input":
					del n.layer[i]
					break

			n.input.extend(['data'])
			n.input_dim.extend([1,getNumChannels(tags_noSize),size,size])
			f.write(str(n))

		#Create snapshot directory
		snapshot_out = os.path.join(out_dir,SNAPSHOT_FOLDER)
		if not os.path.exists(snapshot_out):
			print "Snapshot Directory Not Found, Creating"
			os.makedirs(snapshot_out)


		exp_folder = EXPERIMENTS_FOLDER(ds,group,experiment,exp_num)
		snapshot_solver = os.path.join(exp_folder, SNAPSHOT_FOLDER, experiment)
		train_val_solver = os.path.join(exp_folder, TRAIN_VAL)

		solver = os.path.join(out_dir, SOLVER)
		with open(solver, "w") as f:
			f.write("net: \"%s\"\n" % (train_val_solver))
			if multiple:
				f.write("test_iter: %d\n" % MULTIPLE_TEST_ITERS[ds][tag]) 
			else:
				f.write("test_iter: %d\n" % 1000)  # for all non-multiple datasets
			f.write("snapshot_prefix: \"%s\"\n" % (snapshot_solver))
			for param, val in SOLVER_PARAM.items():
				f.write("%s: %s\n" % (param, str(val)))

			if group != 'finetune2':
				f.write("base_lr: %f\n" % (LEARNING_RATES[ds]))
				f.write("max_iter: %d\n" % (MAX_ITER[ds]))
				f.write("stepsize: %d\n" % (STEP_SIZE[ds]))
			else:
				f.write("base_lr: %f\n" % 0.001)
				f.write("max_iter: %d\n" % 100000)
				f.write("stepsize: %d\n" % 40000)

		

def createEquivarianceExperiment(ds, tags, group, experiment, num_experiments=1, shift='mean', scale=(1.0/255), 
								mapping='linear', l_tparams=[{}], l2_loss_weight=2., ce_loss_weight=1., im_size=227,
								batch_size=8):

	# Check if tags are all the same size or not
	# If they aren't we are doing multi-scale training, and need to stick them all
	# in the same doc data layer 
	if not isinstance(tags, list):
		tags = [tags]

	tags_noSize = map(getTagWithoutSize, tags)
	if shift == "mean":
		shift = map(lambda t: MEAN_VALUES[ds][t], tags_noSize)
	if not isinstance(scale, list):
		scale = len(tags) * [scale]


	for exp_num in range(1,num_experiments+1):
		exp_num = str(exp_num)

		out_dir = OUTPUT_FOLDER(ds, group, experiment, exp_num)
		print out_dir

		if not os.path.exists(out_dir):
			print "Directory Not Found, Creating"
			os.makedirs(out_dir)
		
		sources = map(lambda t: LMDB_PATH(ds, t, "1"), tags)
		sources_tr, sources_val, sources_ts =  zip(*sources)

		#common parameters
		params = dict(sources=list(sources_tr), num_output=OUTPUT_SIZES[ds], shift_channels=shift, scale_channels=scale, 
					batch_size=batch_size, mapping=mapping, l_tparams=l_tparams, total_l2_loss_weight=l2_loss_weight,
					total_non_first_ce_loss_weight=ce_loss_weight)
		params['val_batch_size'] = 4 if ds != 'imagenet' else 5

		#create train_val file
		train_val = os.path.join(out_dir, TRAIN_VAL)
		with open(train_val, "w") as f:
			n = str(createEquivarianceNetwork(val_sources=list(sources_val), **params))
			f.write(re.sub("VAL_", "", n))
	
		#Create train_test file
		train_test = os.path.join(out_dir, TRAIN_TEST)
		with open(train_test, "w") as f:
			n = str(createEquivarianceNetwork(val_sources=list(sources_ts), **params))
			f.write(re.sub("VAL_", "", n))

		#Create Deploy File
		deploy_file = os.path.join(out_dir, DEPLOY_FILE)
		with open(deploy_file, "w") as f:
			n = createEquivarianceNetwork(deploy=True, **params)
			for i, l in enumerate(n.layer):
				if l.type == "Input":
					del n.layer[i]
					break

			n.input.extend(['data'])
			n.input_dim.extend([1,getNumChannels(tags_noSize),im_size,im_size])
			f.write(str(n))

		#Create snapshot directory
		snapshot_out = os.path.join(out_dir,SNAPSHOT_FOLDER)
		if not os.path.exists(snapshot_out):
			print "Snapshot Directory Not Found, Creating"
			os.makedirs(snapshot_out)


		exp_folder = EXPERIMENTS_FOLDER(ds,group,experiment,exp_num)
		snapshot_solver = os.path.join(exp_folder, SNAPSHOT_FOLDER, experiment)
		train_val_solver = os.path.join(exp_folder, TRAIN_VAL)

		solver = os.path.join(out_dir, SOLVER)
		with open(solver, "w") as f:
			f.write("net: \"%s\"\n" % (train_val_solver))
			#f.write("base_lr: %f\n" % 0.003)
			f.write("base_lr: %f\n" % 0.0005)
			#f.write("max_iter: %d\n" % 500000)
			f.write("max_iter: %d\n" % 150000)
			#f.write("stepsize: %d\n" % 150000)
			f.write("stepsize: %d\n" % 60000)
			f.write("test_iter: %d\n" % 10000) 
			f.write("test_interval: %d\n" % 5000) 
			f.write("snapshot: %d\n" % 5000) 
			f.write("lr_policy: \"%s\"\n" % "step") 
			f.write("gamma: %f\n" % 0.1) 
			f.write("display: %d\n" % 100) 
			f.write("momentum: %f\n" % 0.9) 
			f.write("solver_mode: %s\n" % 'GPU') 
			f.write("snapshot_prefix: \"%s\"" % (snapshot_solver))

