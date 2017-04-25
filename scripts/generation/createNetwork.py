#!/usr/bin/python

import argparse
import os
import re

import caffe
from caffe import layers as L
from caffe import params as P
import caffe.proto.caffe_pb2 as proto

import numpy as np
import random
import collections
import lmdb

ROOT="/fslgroup/fslg_icdar/compute"


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
	


def OUTPUT_FOLDER(experiment_set, dataset, group, experiment, split):
	return os.path.join("experiments", experiment_set, "nets" , dataset, group, experiment, split)


def EXPERIMENTS_FOLDER(experiment_set, dataset, group, experiment, split):
	return os.path.join(ROOT, OUTPUT_FOLDER(experiment_set, dataset, group, experiment, split))


def LMDB_PATH(dataset, tag, size, data_partition='train'):
	if not isinstance(tag, basestring):
		path = os.path.join(ROOT, "data/chris", dataset, 'lmdb', str(size), tag[0], "%s_%s_lmdb" % (tag[1], data_partition))
	else:
		path = os.path.join(ROOT, "data/chris", dataset, 'lmdb', str(size), tag, "%s_%s_lmdb" % (tag, data_partition))
	return path
	

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




TRAIN_TRAIN = "train_train.prototxt"
TRAIN_VAL = "train_val.prototxt"
TRAIN_TEST = "train_test.prototxt"
DEPLOY_FILE = "deploy.prototxt"
SOLVER = "solver.prototxt"
SNAPSHOT_FOLDER = "snapshots"

def createTransformParam(scale, shift,  seed, rotate=False, shear=False, perspective=False, 
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
		


def createNetwork(train_input_sources=[], train_label_sources={}, val_input_sources=[], val_label_sources={},
						  outputs=[], depth=3, kernel_size=3, num_filters=24, num_scales=1, lrn=0, pool=0, global_features=0, 
						  later_convs=0, deploy=False, seed=None, rotate=False, shear=False, perspective=False, elastic=False, 
						  color_jitter=False, blur=False, noise=False, zero_border=0, train_batch_size=5, densenet=False, 
						  residual=False, margin=0.5, num_upsample_filters=None, later_layers_kernel_size=3, round_0_weight=0.5,
						  round_0_only=False):
	assert deploy or len(train_input_sources) == len(val_input_sources)
	assert deploy or len(train_label_sources) == len(val_label_sources)
	assert deploy or train_input_sources
	if num_upsample_filters is None:
		num_last_layer_filters = num_filters
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
				seed=seed, rand_advance_skip=10, image_transform_param=createTransformParam(scale=(1./255), shift=127, seed=seed, 
					rotate=rotate, shear=shear, perspective=perspective, elastic=elastic, color_jitter=color_jitter,
					blur=blur, noise=noise))
			inputs.append(input)

		if len(inputs) == 1:
			n.data = inputs[0]
		else:
			n.data = L.Concat(*inputs, include=dict(phase=caffe.TRAIN))

		# training labels
		for output_name, sources in train_label_sources.iteritems():
			gt = L.DocData(sources=[sources[0]], include=dict(phase=caffe.TRAIN), batch_size=train_batch_size, 
				backend=P.Data.LMDB, seed=seed, rand_advance_skip=10,
				image_transform_param=createTransformParam(scale=1, shift=None, seed=seed, gt=True,
					rotate=rotate, shear=shear, perspective=perspective, elastic=elastic, color_jitter=False, blur=False, noise=False))
			setattr(n, "gt_%s" % output_name, gt)

			recall_weights = L.DocData(sources=[sources[1]], include=dict(phase=caffe.TRAIN), batch_size=train_batch_size, 
				backend=P.Data.LMDB, seed=seed, rand_advance_skip=10, image_transform_param=createTransformParam(scale=(1./128), shift=0, 
					seed=seed, gt=True, rotate=rotate, shear=shear, perspective=perspective, elastic=elastic, color_jitter=False, 
					blur=False, noise=False, zero_border=zero_border))
			setattr(n, "recall_weights_%s" % output_name, recall_weights)

			precision_weights = L.DocData(sources=[sources[2]], include=dict(phase=caffe.TRAIN), batch_size=train_batch_size, 
				backend=P.Data.LMDB, seed=seed, rand_advance_skip=10, image_transform_param=createTransformParam(scale=(1./128), shift=0, 
					seed=seed, gt=True, rotate=rotate, shear=shear, perspective=perspective, elastic=elastic, color_jitter=False, 
					blur=False, noise=False, zero_border=zero_border))
			setattr(n, "precision_weights_%s" % output_name, precision_weights)
			
		# val inputs
		inputs = list()

		for source in val_input_sources:
			input = L.DocData(sources=[source], include=dict(phase=caffe.TEST), batch_size=1, backend=P.Data.LMDB,
				image_transform_param=createTransformParam(scale=(1./255), shift=127, seed=seed))
			inputs.append(input)

		if len(inputs) == 1:
			n.VAL_data = inputs[0]
		else:
			n.VAL_data = L.Concat(*inputs, include=dict(phase=caffe.TEST))
			
		# val labels
		for output_name, sources in val_label_sources.iteritems():
			gt = L.DocData(sources=[sources[0]], include=dict(phase=caffe.TEST), batch_size=1, backend=P.Data.LMDB,
				image_transform_param=createTransformParam(scale=1, shift=None, seed=seed, gt=True))
			setattr(n, "VAL_gt_%s" % output_name, gt)

			recall_weights = L.DocData(sources=[sources[1]], include=dict(phase=caffe.TEST), batch_size=1, backend=P.Data.LMDB,
				image_transform_param=createTransformParam(scale=(1./128), shift=0, seed=seed, gt=True))
			setattr(n, "VAL_recall_weights_%s" % output_name, recall_weights)

			precision_weights = L.DocData(sources=[sources[2]], include=dict(phase=caffe.TEST), batch_size=1, backend=P.Data.LMDB,
				image_transform_param=createTransformParam(scale=(1./128), shift=None, seed=seed, gt=True))
			setattr(n, "VAL_precision_weights_%s" % output_name, precision_weights)

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
			scale_kernel_size = max(3, kernel_size - 2 * (scale / 2))
			scale_pad_size = (scale_kernel_size - 1) / 2

			# do the densenet thing
			if densenet and conv_idx > 0:
				prev_layer = L.Concat(*layers[scale])

			if residual and conv_idx > 0:
				residual_layer = convLayer(prev_layer, kernel_size=scale_kernel_size, pad=scale_pad_size, 
					num_output=num_filters, stride=1, lrn=do_lrn)
				prev_layer = L.Eltwise(residual_layer, prev_layer)
			else:
				# create a single conv layer
				prev_layer = convLayer(prev_layer, kernel_size=scale_kernel_size, pad=scale_pad_size, 
					num_output=num_filters, stride=1, lrn=do_lrn)

			# This does max-pooling without downsampling
			if pool > (conv_idx + scale):
				prev_layer = poolLayer(prev_layer, kernel_size=3, stride=1, pad=1)
			layers[scale].append(prev_layer)
			setattr(n, "conv_%d_%d" % (scale, conv_idx), prev_layer)

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
		prev_layer = convLayer(layers[num_scales-1][0], kernel_size=3, pad=1, num_output=num_filters, stride=2)
		prev_layer = L.Pooling(prev_layer, pool=P.Pooling.AVE, global_pooling=True)
		for conv_idx in xrange(global_features): 
			prev_layer = convLayer(prev_layer, kernel_size=1, pad=0, num_output=num_filters, stride=1)
			layers[num_scales].append(prev_layer)
			setattr(n, "conv_global_%d" % conv_idx, prev_layer)

	# collect the last layers in each scale
	last_layers = []
	for scale in xrange(num_scales + (1 if global_features > 0 else 0)):
		scale_layers = layers[scale]
		if scale_layers:
			if scale and num_upsample_filters != num_filters:
				# do a 1x1 conv projection
				last_layer = convLayer(scale_layers[-1], kernel_size=1, pad=0, num_output=num_upsample_filters, stride=1)
			else:
				last_layer = scale_layers[-1]
			last_layers.append(last_layer)
	
	# resize smaller scales to original size
	if len(last_layers) > 1:
		for idx in xrange(len(last_layers)):
			if idx != 0:
				last_layers[idx] = L.BilinearInterpolation(last_layers[idx], n.data)
				setattr(n, "upsample_%d" %  idx, last_layers[idx])
		n.merged = L.Concat(*last_layers)
		n.merged_conv = convLayer(n.merged, kernel_size=kernel_size, pad=pad_size, num_output=num_filters, stride=1)
		layers[0].append(n.merged_conv)
		prev_layer = n.merged_conv

	# apply any number of convolutions
	later_pad = (later_layers_kernel_size - 1) / 2
	for idx in xrange(later_convs):
		prev_layer = convLayer(prev_layer, kernel_size=later_layers_kernel_size, pad=later_pad, num_output=num_filters, stride=1)
		layers[0].append(prev_layer)

	if densenet:
		prev_layer = L.Concat(*layers[0])

	n.rep_layer_0 = prev_layer

	# round 0 of classification
	for output, loss_weight in outputs:
		prev_layer = convLayer(n.rep_layer_0, kernel_size=later_layers_kernel_size, pad=later_pad, num_output=num_filters, stride=1)
		layers[output].append(prev_layer)
		probs = convLayerSigmoid(prev_layer, kernel_size=1, pad=0, num_output=1, stride=1)
		setattr(n, '%s_prob_0' % output, probs)
		layers[output].append(probs)
		if not deploy:
			loss = L.WeightedFmeasureLoss(probs, 
				getattr(n, 'gt_%s' % output), 
				getattr(n, 'recall_weights_%s' % output), 
				getattr(n, 'precision_weights_%s' % output), 
				loss_weight=(round_0_weight * loss_weight / len(outputs)), 
				margin=margin)
			setattr(n, '%s_loss_0' % output, loss)

	if not round_0_only:
		# round 1 of classification
		n.rep_layer_1 = convLayer(n.rep_layer_0, kernel_size=later_layers_kernel_size, pad=later_pad, num_output=num_filters, stride=1)
		prob_layers = [layers[output][-1] for output, _ in outputs]
		n.augmented_rep_layer = L.Concat(n.rep_layer_1, *prob_layers, propagate_down=[True] + len(prob_layers) * [False] )

		for output, loss_weight in outputs:
			prev_layer = convLayer(n.augmented_rep_layer, kernel_size=later_layers_kernel_size, pad=later_pad, num_output=num_filters, stride=1)
			layers[output].append(prev_layer)
			probs = convLayerSigmoid(prev_layer, kernel_size=1, pad=0, num_output=1, stride=1, ntop=1)
			setattr(n, '%s_prob_1' % output, probs)
			layers[output].append(probs)
			if not deploy:
				loss = L.WeightedFmeasureLoss(probs, 
					getattr(n, 'gt_%s' % output), 
					getattr(n, 'recall_weights_%s' % output), 
					getattr(n, 'precision_weights_%s' % output), 
					loss_weight=loss_weight / len(outputs), 
					margin=margin)
				setattr(n, '%s_loss_1' % output, loss)
		
	return n.to_proto()


def createExperiment(exp_set, ds, tags, outputs, group, experiment, num_experiments=1, lr=0.0005, uniform_weights=False,
		input_size=256, train_batch_size=5, **kwargs):

	sources_input_train = map(lambda tag: LMDB_PATH(ds, tag, input_size, data_partition='train'), tags)
	sources_input_val = map(lambda tag: LMDB_PATH(ds, tag, input_size, data_partition='val'), tags)
	sources_input_test = map(lambda tag: LMDB_PATH(ds, tag, input_size, data_partition='test'), tags)

	sources_label_train = dict()
	sources_label_val = dict()
	sources_label_test = dict()
	for output, loss_weight in outputs:
		if uniform_weights:
			label_tags = [output, 'uniform_weights', 'uniform_weights']
		else:
			label_tags = [output, '%s_recall_weights' % output, '%s_precision_weights' % output]

		sources_label_train[output] = map(lambda tag: LMDB_PATH(ds, tag, input_size, data_partition='train'), label_tags)
		sources_label_val[output] = map(lambda tag: LMDB_PATH(ds, tag, input_size, data_partition='val'), label_tags)
		sources_label_test[output] = map(lambda tag: LMDB_PATH(ds, tag, input_size, data_partition='test'), label_tags)

	params = {'train_input_sources': sources_input_train, 'train_label_sources': sources_label_train, 'outputs': outputs,
			  'train_batch_size': train_batch_size}
	params.update(kwargs)
	
	for exp_num in range(1,num_experiments+1):
		exp_num = str(exp_num)

		out_dir = OUTPUT_FOLDER(exp_set, ds, group, experiment, exp_num)
		print out_dir

		if not os.path.exists(out_dir):
			os.makedirs(out_dir)

		#create train_train file
		train_val = os.path.join(out_dir, TRAIN_TRAIN)
		with open(train_val, "w") as f:
			n = str(createNetwork(val_input_sources=sources_input_train, val_label_sources=sources_label_train, **params))
			f.write(re.sub("VAL_", "", n))

		#create train_val file
		train_val = os.path.join(out_dir, TRAIN_VAL)
		with open(train_val, "w") as f:
			n = str(createNetwork(val_input_sources=sources_input_val, val_label_sources=sources_label_val, **params))
			f.write(re.sub("VAL_", "", n))
	
		#Create train_test file
		train_test = os.path.join(out_dir, TRAIN_TEST)
		with open(train_test, "w") as f:
			n = str(createNetwork(val_input_sources=sources_input_test, val_label_sources=sources_label_test, **params))
			f.write(re.sub("VAL_", "", n))

		#Create Deploy File
		deploy_file = os.path.join(out_dir, DEPLOY_FILE)
		with open(deploy_file, "w") as f:
			n = createNetwork(deploy=True, **params)
			for i, l in enumerate(n.layer):
				if l.type == "Input":
					del n.layer[i]
					break

			n.input.extend(['data'])
			n.input_dim.extend([1, get_num_channels(sources_input_train),input_size,input_size])
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

		exp_folder = EXPERIMENTS_FOLDER(exp_set, ds, group, experiment, exp_num)
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
			f.write("min_iters_per_period: %d\n" % 10000)
			f.write("min_iters_per_period: %d\n" % 2000)
			f.write("min_iters_per_period: %d\n" % 2000)
			f.write("min_lr: %f\n" % 1e-6)
			f.write("max_iter: %d\n" % 200000)
			f.write("max_nonfinite_test_loss: %d\n" % 1)
			f.write("clip_gradients: %f\n" % 10.)

			f.write("test_iter: %d\n" % lmdb_num_entries(sources_label_val.values()[0][0]))
			f.write("test_interval: %d\n" % (lmdb_num_entries(sources_label_train.values()[0][0]) / (2 * train_batch_size)))
			f.write("snapshot: %d\n" % (lmdb_num_entries(sources_label_train.values()[0][0]) / (2 * train_batch_size)))
			f.write("momentum: %f\n" % 0.75)
			f.write("weight_decay: %f\n" % 0.0005)

			f.write("display: %d\n" % 100)
			f.write("solver_mode: GPU\n")
			f.write("snapshot_prefix: \"%s\"" % (snapshot_solver))
	

