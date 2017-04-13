import os
import createNetwork

DS = ['hisdb']
	

########################
#datasets = ['hisdb']
datasets = ['cbad_simple', 'cbad_complex']
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
	group = "arch"
	tags = TAG_SETS["gray"]
	#outputs = [ ('baselines_7', 1.), ('boundary_mask', 0.5), ('comment_mask', 1.), 
	#			('inverted_background_mask', 1.), ('decoration_mask', 1.), ('text_mask', 1.)  ]
	outputs = [ ('baselines_7', 1.) ]

	for depth in [5, 7]:
		for num_filters in [8, 16]:
			for kernel_size in [5, 7]:
				for scale in [1,4]:
					name = "arch_%d_%d_%d_%d" % (depth, num_filters, kernel_size, scale)
					print "createExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
					createNetwork.createExperiment(ds, ds, tags, outputs, group, name, num_experiments=3, uniform_weights=True, 
						kernel_size=kernel_size, depth=depth, num_filters=num_filters, num_scales=scale, train_batch_size=5, 
						input_size=512, num_upsample_filters=8)

def outputExperiments(ds):
	group = "output"
	tags = TAG_SETS["gray"]
	outputs = [ ('baselines_7', 1.), ('boundary_mask', 0.5), ('comment_mask', 1.), 
				('inverted_background_mask', 1.), ('decoration_mask', 1.), ('text_mask', 1.),
				('task3_gt', 1.), ('otsu', 1.)]

	for output in outputs:
		name = "output_%s" % output[0]
		print "createExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
		createNetwork.createExperiment(ds, ds, tags, [output], group, name, num_experiments=3, uniform_weights=True, 
			kernel_size=7, depth=5, num_filters=24, num_scales=3, train_batch_size=10, 
			input_size=512, num_upsample_filters=8)


def outputSingleExperiments(ds):
	group = "output_single"
	tags = TAG_SETS["gray"]
	outputs = [ ('baselines_7', 1.), ('boundary_mask', 0.5), ('comment_mask', 1.), 
				('inverted_background_mask', 1.), ('decoration_mask', 1.), ('text_mask', 1.),
				('task3_gt', 1.), ('otsu', 1.)]

	#for output in outputs:
	#	name = "output_%s" % output[0]
	#	print "createHisDbExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
	#	createNetwork.createHisDbExperiment(ds, tags, [output], group, name, num_experiments=3, uniform_weights=True, 
	#		kernel_size=7, depth=5, num_filters=24, num_scales=3, train_batch_size=10, 
	#		input_size=512, num_upsample_filters=8, round_0_only=True, round_0_weight=1.)

	name = "output_all" 
	print "createHisDbExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
	createNetwork.createExperiment(ds, ds, tags, outputs, group, name, num_experiments=3, uniform_weights=True, 
		kernel_size=7, depth=5, num_filters=16, num_scales=3, train_batch_size=10, 
		input_size=512, num_upsample_filters=8, round_0_only=True, round_0_weight=1.)


if __name__ == "__main__":
	for ds in datasets:
		#augmentExperiments(ds)
		archExperiments(ds)
		#outputExperiments(ds)
		#outputSingleExperiments(ds)

