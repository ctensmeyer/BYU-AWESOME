import os
import createNetwork

	

########################
#datasets = ['hdibco_gray']
#datasets = ['hisdb']
datasets = ['cbad_simple', 'cbad_complex']
#datasets += ['cbad_combined_simple', 'cbad_combined_complex']
#######################

TAG_SETS = {
			"original": ['original_images'],
			"gray": ['gray_images'],
			"singles": ['bilateral', 'percentile', 'otsu', 'wolf'],
			"wide_window": [('mean', 'mean_9'), ('mean', 'mean_19'), ('mean', 'mean_39'), ('mean', 'mean_79'),
			               ('median', 'median_9'), ('median', 'median_19'), ('median', 'median_39'), ('median', 'median_79') ],
			"narrow_window": [('min', 'min_3'), ('min', 'min_5'), ('min', 'min_7'), ('min', 'min_9'),
							('max', 'max_3'), ('max', 'max_5'), ('max', 'max_7'), ('max', 'max_9'),
							('percentile_10', 'percentile_10_3'), ('percentile_10', 'percentile_10_5'), ('percentile_10', 'percentile_10_7'), ('percentile_10', 'percentile_10_9'),
							('percentile_25', 'percentile_25_3'), ('percentile_25', 'percentile_25_5'), ('percentile_25', 'percentile_25_7'), ('percentile_25', 'percentile_25_9'),
							('std_dev', 'std_dev_3'), ('std_dev', 'std_dev_5'), ('std_dev', 'std_dev_7'), ('std_dev', 'std_dev_9')],
			"relative_darkness": [],
			"relative_darkness2": [('relative_darkness2/5/10', 'relative_darkness2_5_10_%s' % x) for x in ['lower', 'middle', 'upper']],
			"color": ['color'],
			"color2": [('slice', 'slice_b'), ('slice', 'slice_g'), ('slice', 'slice_r')],
			"canny": [],
			"howe": ['howe'],
			"otsu": ['otsu'],
			"fcn_bin": ['fcn_binary_images'],
}
#for x in [5, 7, 9]:
#	for y in [10, 20, 40]:
#		TAG_SETS["relative_darkness"].append( ('relative_darkness/%d' % x, 'relative_darkness_%d_%d' % (x, y)) )
#
#for low in [75, 100, 125]:
#	for high in [150, 175, 200]:
#		TAG_SETS["canny"].append( ('canny/%d' % low, 'canny_%d_%d' % (low, high)) )

TAG_SETS["all"] = sum(TAG_SETS.values(), [])


def archExperiments(ds):
	group = "arch384_total_color2"
	tags = TAG_SETS["color2"]
	#outputs = [ ('baselines_7', 1.), ('boundary_mask', 0.5), ('comment_mask', 1.), 
	#			('inverted_background_mask', 1.), ('decoration_mask', 1.), ('text_mask', 1.)  ]
	outputs = [ ('baselines_7', 1.) ]
	round_weights = [0.5, 1]

	for depth in [7]:
		for num_filters in [48]:
			for kernel_size in [7]:
				for scale in [4]:
					for mse_lambda in [0.1]:
						name = "arch_%d_%d_%d_%d_%f" % (depth, num_filters, kernel_size, scale, mse_lambda)
						print "createExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
						createNetwork.createExperiment(ds, ds, tags, outputs, group, name, num_experiments=3,
							kernel_size=kernel_size, depth=depth, num_filters=num_filters, num_scales=scale, train_batch_size=5, 
							input_size=384, num_upsample_filters=num_filters/2, mse_lambda=mse_lambda, avg_fm=False,
							round_weights=round_weights)


def outputPairExperiments(ds):
	group = "output_pair"
	tags = TAG_SETS["color2"]
	outputs = { 'base': ('baselines_7', 1., 4), 'bound': ('boundary_mask', 0.5), 
				'comment': ('comment_mask', 1.), 'back': ('inverted_background_mask', 1.), 
				'dec': ('decoration_mask', 1.), 'text': ('text_mask', 1.), 'task3': ('task3_gt', 1.)}

	pairs = [ ('text', 'comment'), ('text', 'comment', 'base'), ('dec', 'text'), ('text', 'task3'),
			  ('text', 'bound'), ('text', 'base'), ('dec', 'comment'), ('back', 'comment'), ('bound', 'base'),
			  ('comment', 'base'), ('task3', 'base')]

	for pair in pairs:
		name = "output_%s" % "_".join(pair)
		print "createExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
		output = [outputs[p] for p in pair]
		createNetwork.createExperiment(ds, ds, tags, output, group, name, num_experiments=3,
			kernel_size=9, depth=7, num_filters=64, num_scales=4, train_batch_size=5, 
			input_size=256, num_upsample_filters=32, mse_lambda=0, avg_fm=False)


def nwayExperiments(ds):
	group = "nway_round"
	tags = TAG_SETS["color2"]
	output = [ ('original_classes', 1., 10) ]

	for num_rounds in xrange(2,4):
		name = "nway_round_%d" % num_rounds 
		round_weights = [2 ** -(num_rounds - idx - 1) for idx in xrange(num_rounds)]
		createNetwork.createExperiment(ds, ds, tags, output, group, name, num_experiments=3,
			kernel_size=9, depth=7, num_filters=64, num_scales=4, train_batch_size=5, 
			input_size=256, num_upsample_filters=32, round_weights=round_weights, 
			backprop_probs=False, recurrent_rounds=False)
				


def outputExperiments(ds):
	group = "output"
	tags = TAG_SETS["color2"]
	#outputs = [ ('baselines_7', 1.), ('boundary_mask', 0.5), ('comment_mask', 1.), 
	#			('inverted_background_mask', 1.), ('decoration_mask', 1.), ('text_mask', 1.),
	#			('task3_gt', 1.)]

	for output in outputs:
		name = "output_%s" % output[0]
		print "createExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
		createNetwork.createExperiment(ds, ds, tags, [output], group, name, num_experiments=3,
			kernel_size=9, depth=7, num_filters=48, num_scales=4, train_batch_size=5, 
			input_size=256, num_upsample_filters=16, mse_lambda=0, avg_fm=False)

	name = "output_all" 
	print "createExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
	createNetwork.createExperiment(ds, ds, tags, outputs, group, name, num_experiments=3,
		kernel_size=9, depth=7, num_filters=48, num_scales=4, train_batch_size=5, 
		input_size=256, num_upsample_filters=16, mse_lambda=0, avg_fm=False)


def augmentationBaselineExperiments(ds):
	group = "augmentation"
	tags = TAG_SETS["color2"]
	outputs = [ ('baselines_7', 1., 'baselines_7_recall_weights', 'baselines_7_precision_weights') ]

	for aug in ['shear', 'perspective', 'elastic', 'blur', 'noise', 'rotate', 'color_jitter']:
		name = "aug_%s" % aug	
		print "createExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
		d = {aug: True}
		createNetwork.createExperiment(ds, ds, tags, outputs, group, name, num_experiments=3,
			kernel_size=7, depth=7, num_filters=32, num_scales=5, train_batch_size=5, iter_size=1,
			input_size=384, num_upsample_filters=32, lr=0.001, mse_lambda=0, avg_fm=False, **d)


def featuresBaselineExperiments(ds):
	group = "binarization"
	outputs = [ ('baselines_7', 1., 'baselines_7_recall_weights', 'baselines_7_precision_weights') ]

	tags = TAG_SETS["color2"]
	name = "bin_none"
	print "createExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
	createNetwork.createExperiment(ds, ds, tags, outputs, group, name, num_experiments=3,
		kernel_size=9, depth=7, num_filters=32, num_scales=5, train_batch_size=5, iter_size=1,
		input_size=384, num_upsample_filters=32, lr=0.001, mse_lambda=0, avg_fm=False)

	tags = TAG_SETS["color2"] + TAG_SETS["otsu"]
	name = "bin_otsu"
	print "createExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
	createNetwork.createExperiment(ds, ds, tags, outputs, group, name, num_experiments=3,
		kernel_size=9, depth=7, num_filters=32, num_scales=5, train_batch_size=5, iter_size=1,
		input_size=384, num_upsample_filters=32, lr=0.001, mse_lambda=0, avg_fm=False)

	tags = TAG_SETS["color2"] + TAG_SETS["fcn_bin"]
	name = "bin_fcn"
	print "createExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
	createNetwork.createExperiment(ds, ds, tags, outputs, group, name, num_experiments=3,
		kernel_size=9, depth=7, num_filters=32, num_scales=5, train_batch_size=5, iter_size=1,
		input_size=384, num_upsample_filters=32, lr=0.001, mse_lambda=0, avg_fm=False)

	tags = TAG_SETS["color2"] + TAG_SETS["otsu"] + TAG_SETS["fcn_bin"]
	name = "bin_both"
	print "createExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
	createNetwork.createExperiment(ds, ds, tags, outputs, group, name, num_experiments=3,
		kernel_size=9, depth=7, num_filters=32, num_scales=5, train_batch_size=5, iter_size=1,
		input_size=384, num_upsample_filters=32, lr=0.001, mse_lambda=0, avg_fm=False)


def roundBaselineExperiments(ds):
	group = "base_weights_back"
	tags = TAG_SETS["color2"]
	outputs = [ ('baselines_7', 1., 'baselines_7_recall_weights', 'baselines_7_precision_weights') ]

	for num_rounds in xrange(1,4):
		name = "round_weighted_%d" % num_rounds	
		round_weights = [2 ** -(num_rounds - idx - 1) for idx in xrange(num_rounds)]
		print "createExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
		createNetwork.createExperiment(ds, ds, tags, outputs, group, name, num_experiments=3,
			kernel_size=7, depth=7, num_filters=32, num_scales=5, train_batch_size=5, iter_size=1,
			input_size=384, num_upsample_filters=32, lr=0.001, mse_lambda=0, avg_fm=False,
			round_weights=round_weights, backprop_probs=True, recurrent_rounds=False)

	#outputs = [ ('baselines_7', 1.) ]

	#for num_rounds in xrange(1,4):
	#	name = "round_unweighted_%d" % num_rounds	
	#	round_weights = [2 ** -(num_rounds - idx - 1) for idx in xrange(num_rounds)]
	#	print "createExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
	#	createNetwork.createExperiment(ds, ds, tags, outputs, group, name, num_experiments=3,
	#		kernel_size=7, depth=7, num_filters=32, num_scales=5, train_batch_size=5, iter_size=1,
	#		input_size=384, num_upsample_filters=32, lr=0.001, mse_lambda=0, avg_fm=False,
	#		round_weights=round_weights, backprop_probs=False, recurrent_rounds=False)

def hdibcoRoundExperiments(ds):
	group = "round_back"
	tags = TAG_SETS["original"] + TAG_SETS["relative_darkness2"] + TAG_SETS["howe"]
	outputs = [('processed_gt', 1., 'recall_weights', 'precision_weights')]

	for num_rounds in [9]:
		name = "round_%d" % num_rounds	
		round_weights = [2 ** -(num_rounds - idx - 1) for idx in xrange(num_rounds)]
		print "createExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
		createNetwork.createExperiment(ds, ds, tags, outputs, group, name, num_experiments=5,
			kernel_size=9, depth=7, num_filters=64, num_scales=4, train_batch_size=5, iter_size=2,
			input_size=256, num_upsample_filters=32, lr=0.001, mse_lambda=0, avg_fm=False,
			round_weights=round_weights, backprop_probs=True, recurrent_rounds=False, skip_zero_loss=False)


def hdibcoRoundControlExperiments(ds):
	group = "round_control"
	tags = TAG_SETS["original"] + TAG_SETS["relative_darkness2"] + TAG_SETS["howe"]
	outputs = [('processed_gt', 1., 'recall_weights', 'precision_weights')]

	for num_rounds in [2, 5, 9]:
		name = "round_%d" % num_rounds	
		round_weights = [0] * (num_rounds - 1) + [1]
		print "createExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
		createNetwork.createExperiment(ds, ds, tags, outputs, group, name, num_experiments=5,
			kernel_size=9, depth=7, num_filters=64, num_scales=4, train_batch_size=5, iter_size=2,
			input_size=256, num_upsample_filters=32, lr=0.001, mse_lambda=0, avg_fm=False,
			round_weights=round_weights, backprop_probs=False, recurrent_rounds=False, skip_zero_loss=False)

		name = "round_skip_%d" % num_rounds	
		round_weights = [0] * (num_rounds - 1) + [1]
		print "createExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
		createNetwork.createExperiment(ds, ds, tags, outputs, group, name, num_experiments=5,
			kernel_size=9, depth=7, num_filters=64, num_scales=4, train_batch_size=5, iter_size=2,
			input_size=256, num_upsample_filters=32, lr=0.001, mse_lambda=0, avg_fm=False,
			round_weights=round_weights, backprop_probs=False, recurrent_rounds=False, skip_zero_loss=True)

		name = "later_convs_%d" % num_rounds	
		round_weights = [1]
		print "createExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
		createNetwork.createExperiment(ds, ds, tags, outputs, group, name, num_experiments=5,
			kernel_size=9, depth=7, num_filters=64, num_scales=4, train_batch_size=5, iter_size=2,
			input_size=256, num_upsample_filters=32, lr=0.001, mse_lambda=0, avg_fm=False,
			round_weights=round_weights, backprop_probs=False, recurrent_rounds=False, skip_zero_loss=True,
			later_convs=num_rounds-1)

def hdibcoExperiments(ds):
	group = "output2"
	tags = TAG_SETS["original"] + TAG_SETS["relative_darkness2"] + TAG_SETS["howe"]
	base_outputs = [('processed_gt', 1., 'recall_weights', 'precision_weights')]
	outputs = [ #('processed_gt', 1., 'recall_weights', 'precision_weights'), 
				('left_edges', 0.05), ('right_edges', 0.05), ('top_edges', 0.05), 
				('bottom_edges', 0.05), ('convex_hull', 0.05), ('skeleton', 0.05), 
				('dilated_skeleton', 0.05), ('holes', 0.05)]

	for output in outputs:
		name = "output_%s" % output[0]
		output = base_outputs + [output]
		print "createExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
		createNetwork.createExperiment(ds, ds, tags, output, group, name, num_experiments=3,
			kernel_size=9, depth=7, num_filters=64, num_scales=4, train_batch_size=10, 
			input_size=256, num_upsample_filters=32, lr=0.001)

	#name = "output_all" 
	#print "createExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
	#createNetwork.createExperiment(ds, ds, tags, outputs, group, name, num_experiments=3,
	#	kernel_size=9, depth=7, num_filters=32, num_scales=4, train_batch_size=10, 
	#	input_size=256, num_upsample_filters=16, lr=0.001)


def outputSingleExperiments(ds):
	group = "output_single_256"
	tags = TAG_SETS["gray"]
	outputs = [ ('baselines_7', 1.), ('boundary_mask', 0.5), ('comment_mask', 1.), 
				('inverted_background_mask', 1.), ('decoration_mask', 1.), ('text_mask', 1.),
				('task3_gt', 1.)]

	for output in outputs:
		name = "output_%s" % output[0]
		print "createExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
		createNetwork.createExperiment(ds, ds, tags, [output], group, name, num_experiments=3,
			kernel_size=9, depth=7, num_filters=24, num_scales=5, train_batch_size=5, 
			input_size=256, num_upsample_filters=8, round_0_only=True, round_0_weight=1.)

	name = "output_all" 
	print "createExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
	createNetwork.createExperiment(ds, ds, tags, outputs, group, name, num_experiments=3,
		kernel_size=9, depth=7, num_filters=24, num_scales=5, train_batch_size=5, 
		input_size=256, num_upsample_filters=8, round_0_only=True, round_0_weight=1.)


if __name__ == "__main__":
	for ds in datasets:
		#archExperiments(ds)
		#outputExperiments(ds)
		#nwayExperiments(ds)
		#outputSingleExperiments(ds)
		#roundBaselineExperiments(ds)
		featuresBaselineExperiments(ds)
		#augmentationBaselineExperiments(ds)
		#outputPairExperiments(ds)
		#hdibcoExperiments(ds)
		#hdibcoRoundExperiments(ds)
		#hdibcoRoundControlExperiments(ds)

