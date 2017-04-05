import os
import createNetwork

DS = ['hisdb']
	

########################
datasets = ['hisdb']
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
for x in [5, 7, 9]:
	for y in [10, 20, 40]:
		TAG_SETS["relative_darkness"].append( ('relative_darkness/%d' % x, 'relative_darkness_%d_%d' % (x, y)) )

for low in [75, 100, 125]:
	for high in [150, 175, 200]:
		TAG_SETS["canny"].append( ('canny/%d' % low, 'canny_%d_%d' % (low, high)) )

TAG_SETS["all"] = sum(TAG_SETS.values(), [])

def	howeFeaturesExperiments(ds):
	group = "howe"
	tags = TAG_SETS["original"] + ["howe"]

	name = "howe"
	print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':4})
	createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=4, depth=4, kernel_size=7, wfm_loss=True)

def oneConvExperiments(ds):
	group = "conv1s"
	tags = TAG_SETS["original"]

	for one_convs in [0, 1, 2]:
		name = "conv1_%d" % one_convs
		print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3})
		createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=True, one_convs=one_convs)


def lrnExperiments(ds):
	group = "lrn"
	tags = TAG_SETS["original"]

	for do_lrn in [0, 1, 2, 3, 4]:
		name = "lrn_%s" % do_lrn
		print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3})
		createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=True, lrn=do_lrn)


def zeroExperiments(ds):
	group = "zero"
	tags = TAG_SETS["original"]

	for zero in [0, 3, 6, 9, 12, 15]:
		name = "zero_%s" % zero
		print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3})
		createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=True, zero_border=zero)


def poolExperiments(ds):
	group = "pool"
	tags = TAG_SETS["original"]

	for pool in [0, 1, 2, 3, 4]:
		name = "pool_%s" % pool
		print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3})
		createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=True, pool=pool)
		

def batchSizeExperiments(ds):
	group = "batch"
	tags = TAG_SETS["original"]

	for batch_size in [1, 2, 3, 4, 6, 8, 10]:
		name = "batch_%s" % batch_size
		print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':5})
		createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=5, depth=7, kernel_size=9, 
			num_filters=64, wfm_loss=True, train_batch_size=batch_size)


def lrExperiments(ds):
	group = "lr"
	tags = TAG_SETS["original"]

	for lr in [0.0075, 0.005, 0.0025, 0.001, 0.0005, 0.00001]:
		name = "lr_%s" % lr
		print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':5})
		createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=5, depth=7, kernel_size=9, 
			num_filters=64, wfm_loss=True, train_batch_size=5, lr=lr)
	

def augmentExperiments(ds):
	group = "augment"
	tags = TAG_SETS["original"]

	for augmentation in ['rotate', 'shear', 'perspective', 'color_jitter', 'elastic', 'blur', 'noise']:
		name = "augment_%s" % augmentation
		print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':5, augmentation: True})
		kwargs = {augmentation: True}
		createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=5, depth=7, kernel_size=9, 
			num_filters=64, wfm_loss=True, **kwargs)


def augment2Experiments(ds):
	group = "augment2"
	tags = TAG_SETS["original"]

	for augmentation in ['rotate', 'shear', 'color_jitter', 'elastic', 'blur', 'noise']:
		name = "augment_%s" % augmentation
		print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':10, augmentation: True})
		kwargs = {augmentation: True}
		createNetwork.createHisDbExperiment(ds, tags, group, name, train_batch_size=5, num_experiments=10, depth=7, kernel_size=9, 
			num_filters=64, wfm_loss=True, **kwargs)

	createNetwork.createHisDbExperiment(ds, tags, group, "baseline", train_batch_size=5, num_experiments=10, depth=7, kernel_size=9, 
		num_filters=64, wfm_loss=True)


def augment3Experiments(ds):
	group = "augment3"
	tags = TAG_SETS["original"]

	for augmentation in ['color_jitter', 'blur']:
		name = "augment_%s" % augmentation
		print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':10, augmentation: True})
		kwargs = {augmentation: True}
		createNetwork.createHisDbExperiment(ds, tags, group, name, train_batch_size=5, num_experiments=10, depth=7, kernel_size=9, 
			num_filters=48, num_scales=4, wfm_loss=True, **kwargs)

	name = "augment_both"
	print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':10, 'blur': True, 'color_jitter': True})
	kwargs = {augmentation: True}
	createNetwork.createHisDbExperiment(ds, tags, group, name, train_batch_size=5, num_experiments=10, depth=7, kernel_size=9, 
		num_filters=48, num_scales=4, wfm_loss=True, color_jitter=True, blur=True)

	createNetwork.createHisDbExperiment(ds, tags, group, "baseline", train_batch_size=5, num_experiments=10, depth=7, kernel_size=9, 
		num_filters=48, num_scales=4, wfm_loss=True)


def loss2Experiments(ds):
	group = "loss2"
	tags = TAG_SETS["original"]

	# Uniform F-measure
	for loss_weight in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1, 1.5, 2]:
		name = "loss_uniform_fmeasure_%s" % loss_weight
		print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, "wfm_loss": True, 'uniform_weights': True})
		createNetwork.createHisDbExperiment(ds, tags, group, name, train_batch_size=5, num_experiments=3, depth=7, kernel_size=9, wfm_loss=True, num_filters=64, uniform_weights=True, pfm_loss_weight=loss_weight)


def loss3Experiments(ds):
	group = "loss3"
	tags = TAG_SETS["original"]

	# different recall weights
	for recall_tag in ['dilated', 'modified']:
		name = "loss_psuedo_fmeasure_%s" % recall_tag
		print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, "wfm_loss": True})
		createNetwork.createHisDbExperiment(ds, tags, group, name, train_batch_size=5, num_experiments=3, depth=7, kernel_size=9, wfm_loss=True, num_filters=64, recall_weights=recall_tag)

def loss4Experiments(ds):
	group = "loss4"
	tags = TAG_SETS["original"]

	# different recall weights
	for margin in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5]:
		name = "loss_margin_%f" % margin
		print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, "wfm_loss": True})
		createNetwork.createHisDbExperiment(ds, tags, group, name, train_batch_size=5, num_experiments=3, depth=7, kernel_size=9, wfm_loss=True, num_filters=64, margin=margin)


def lossExperiments(ds):
	group = "loss"
	tags = TAG_SETS["original"]

	# Pseudo F-measure
	name = "loss_psuedo_fmeasure"
	print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, 'wfm_loss': True})
	createNetwork.createHisDbExperiment(ds, tags, group, name, train_batch_size=5, num_experiments=3, depth=7, kernel_size=9, num_filters=64, wfm_loss=True)

	# Modified Pseudo F-measure
	for recall_shift in [0.1, 0.25, 0.5, 1]:
		name = "loss_modified_psuedo_fmeasure_%f" % recall_shift
		print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, 'wfm_loss': True})
		createNetwork.createHisDbExperiment(ds, tags, group, name, train_batch_size=5, num_experiments=3, depth=7, kernel_size=9, num_filters=64, wfm_loss=True, recall_shift=recall_shift)

	# Uniform F-measure
	name = "loss_uniform_fmeasure"
	print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, "wfm_loss": True, 'uniform_weights': True})
	createNetwork.createHisDbExperiment(ds, tags, group, name, train_batch_size=5, num_experiments=3, depth=7, kernel_size=9, num_filters=64, wfm_loss=True, uniform_weights=True)

	# Sigmoid Cross Entropy
	name = "loss_cross_entropy"
	print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, "wfm_loss": False})
	createNetwork.createHisDbExperiment(ds, tags, group, name, train_batch_size=5, num_experiments=3, depth=7, kernel_size=9, num_filters=64, wfm_loss=False)


def depthExperiments(ds):
	group = "depth"
	tags = TAG_SETS["original"]

	for depth in [0, 1, 2, 3, 4, 5]:
		name = "depth_%d" % depth
		print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, 'depth':depth})
		createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=3, kernel_size=5, depth=depth, wfm_loss=True)


def arch2Experiments(ds):
	group = "arch2"
	tags = TAG_SETS["gray"]

	for depth in [7,9,11]:
		for num_filters in [24]:
			for kernel_size in [7, 9, 11, 13]:
				for scale in [4, 5, 6]:
					name = "arch_%d_%d_%d_%d" % (depth, num_filters, kernel_size, scale)
					print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, 'depth':depth})
					createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=3, kernel_size=kernel_size, 
						depth=depth, num_filters=num_filters, wfm_loss=True, num_scales=scale, train_batch_size=5, input_size=256,
						baseline_width=7)


def arch3Experiments(ds):
	group = "arch3"
	tags = TAG_SETS["original"]

	depth = 7
	for num_filters in [48, 64, 80]:
		for kernel_size in [9, 11, 13]:
			name = "arch_%d_%d_%d" % (depth, num_filters, kernel_size)
			print "createHisDbExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
			createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=5, kernel_size=kernel_size, 
				depth=depth, num_filters=num_filters, wfm_loss=True, train_batch_size=5)


def archExperiments(ds):
	group = "arch"
	#tags = TAG_SETS["original"]

	for tags in [TAG_SETS['original'], TAG_SETS['gray']]:
		for depth in [1,3,5,7]:
			for num_filters in [8,16,24]:
				for kernel_size in [5,7,9]:
					for scale in [1,4]:
						if scale >= depth:
							continue
						for baseline_width in [1,3,5,7]:
							name = "arch_%s_%d_%d_%d_%d_%d" % (tags[0], depth, num_filters, kernel_size, scale, baseline_width)
							print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':1, 'depth':depth})
							createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=1, kernel_size=kernel_size, 
								depth=depth, num_filters=num_filters, wfm_loss=True, num_scales=scale, train_batch_size=3, baseline_width=baseline_width)

def arch512Experiments(ds):
	group = "arch512"
	#tags = TAG_SETS["original"]

	for tags in [TAG_SETS['original'], TAG_SETS['gray']]:
		for depth in [1,3,5,7]:
			for num_filters in [8,16,24]:
				for kernel_size in [5,7,9]:
					for scale in [4]:
						if scale >= depth:
							continue
						for baseline_width in [1,3,5,7]:
							name = "arch_%s_%d_%d_%d_%d_%d" % (tags[0], depth, num_filters, kernel_size, scale, baseline_width)
							print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':1, 'depth':depth})
							createNetwork.createHisDbExperiment(ds, tags, group, name, input_size=512, num_experiments=1, 
								kernel_size=kernel_size, depth=depth, num_filters=num_filters, wfm_loss=True, num_scales=scale, 
								train_batch_size=3, baseline_width=baseline_width)

def finalExperiments(ds):
	group = "final"
	tags = TAG_SETS["original"]
	final_params = {'num_experiments':10, 'kernel_size':9, 'num_scales': 4, 'num_filters': 64, 'color_jitter': True,
					'train_batch_size': 10}

	name = "final"
	print "createHisDbExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
	createNetwork.createHisDbExperiment(ds, tags, group, name, wfm_loss=True, **final_params) 


def residualExperiments(ds):
	group = "residual_arch"
	tags = TAG_SETS["original"]

	for depth in [4, 5, 6, 7]:
		for num_filters in [24, 48, 64, 80]:
			for kernel_size in [7, 9]:
				name = "arch_%d_%d_%d" % (depth, num_filters, kernel_size)
				print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, 'depth':depth})
				createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=3, kernel_size=kernel_size, 
					depth=depth, num_filters=num_filters, wfm_loss=True, train_batch_size=5, residual=True)

def denseExperiments(ds):
	group = "dense_arch"
	tags = TAG_SETS["original"]

	for depth in [4, 5, 6, 7]:
		for num_filters in [4, 8, 12, 16, 20]:
			for kernel_size in [5, 7, 9]:
				name = "arch_%d_%d_%d" % (depth, num_filters, kernel_size)
				print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, 'depth':depth})
				createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=3, kernel_size=kernel_size, 
					depth=depth, num_filters=num_filters, wfm_loss=True, train_batch_size=5, densenet=True)



def widthExperiments(ds):
	group = "width"
	tags = TAG_SETS["original"]

	for width in [6, 12, 24, 36, 48, 64, 96, 128]:
		name = "width_%d" % width
		print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, 'width':width})
		createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=True, num_filters=width)


def kernelSizeExperiments(ds):
	group = "kernel_size"
	tags = TAG_SETS["original"]


	for size in [3, 5, 7, 9, 11]:
		name = "kernel_size_%d" % size
		print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, 'kernel_size':size})
		createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=3, kernel_size=size, depth=5, wfm_loss=True)


def scaleExperiments(ds):
	group = "scale"
	tags = TAG_SETS["original"]

	for scale in [2, 3, 4]:
		for _global in [0, 2]:
			for kernel_size in [3, 5, 7]:
				name = "scale_%d_kernel_%d_global_%d" % (scale, kernel_size,_global)
				print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, 'scale':scale, 'global_features':_global})
				createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=3, depth=7, wfm_loss=True, 
					kernel_size=kernel_size, num_filters=64, train_batch_size=5, num_scales=scale, global_features=_global)


def scale2Experiments(ds):
	group = "scale2"
	tags = TAG_SETS["original"]

	for scale in [3, 4]:
		for _global in [0, 2]:
			for kernel_size in [7, 9]:
				name = "scale_%d_kernel_%d_global_%d" % (scale, kernel_size,_global)
				print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':10, 'scale':scale, 'global_features':_global})
				createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=10, depth=7, wfm_loss=True, 
					kernel_size=kernel_size, num_filters=64, train_batch_size=5, num_scales=scale, global_features=_global)


def scale3Experiments(ds):
	group = "scale3"
	tags = TAG_SETS["original"]

	for num_filters in [24, 36, 48, 64, 80, 96]:
		name = "scale_%d" % num_filters
		print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':10})
		createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=10, depth=7, wfm_loss=True, 
			kernel_size=9, num_filters=num_filters, train_batch_size=5, num_scales=4)


def scale4Experiments(ds):
	group = "scale4"
	tags = TAG_SETS["original"]

	for kernel_size in [9, 11, 13]:
		name = "scale_%d" % kernel_size
		print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':10})
		createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=10, depth=7, wfm_loss=True, 
			kernel_size=kernel_size, num_filters=48, train_batch_size=5, num_scales=4, color_jitter=True)


def features2Experiments(ds):
	group = "features2"
	tags = TAG_SETS["original"] + [('canny/75', 'canny_75_200'), ('canny/100', 'canny_100_200'), ('canny/75', 'canny_75_175')]
	name = "features_canny"
	print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':5})
	createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=5, train_batch_size=5, depth=7, kernel_size=9, num_filters=64, num_scales=4, wfm_loss=True)

def features3Experiments(ds):
	group = "features3"
	tags = TAG_SETS["original"] + TAG_SETS["relative_darkness2"]
	name = "features_rd_3"
	print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':5})
	createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=5, train_batch_size=5, depth=7, kernel_size=9, num_filters=64, num_scales=4, wfm_loss=True)

	tags = TAG_SETS["original"] + TAG_SETS["color"]
	name = "features_gray_color"
	print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':5})
	createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=5, train_batch_size=5, depth=7, kernel_size=9, num_filters=64, num_scales=4, wfm_loss=True)


def featuresExperiments(ds):
	group = "features"
	base_tags = TAG_SETS["original"]


	for additional_tag in TAG_SETS['all']:
		if isinstance(additional_tag, basestring):
			name = "features_%s" % additional_tag
		else:
			name = "features_%s" % additional_tag[1]
		tags = base_tags + [additional_tag]
		print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':5})
		createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=5, train_batch_size=5, depth=7, kernel_size=9, num_filters=64, num_scales=4, wfm_loss=True)


def channel2Experiments(ds):
	group = "channel2"
	base_tags = TAG_SETS["original"]

	# combined narrow windows
	for size in [3, 5, 7, 9]:
		tags = [(feature, '%s_%d' % (feature, size)) for feature in ['min', 'max', 'percentile_10', 'percentile_25', 'std_dev']]
		tags += [ ('relative_darkness/%d' % size,  'relative_darkness_%d_%d' % (size, thresh)) for thresh in [10, 20]]
		tags += base_tags
		name = "channel_narrow_%d" % size
		print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3})
		createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=True)

	# combined narrow windows with wide_windows
	for narrow_size, wide_size  in zip([3, 5, 7, 9], [9, 19, 39, 79]):
		tags = [(feature, '%s_%d' % (feature, narrow_size)) for feature in ['min', 'max', 'percentile_10', 'percentile_25', 'std_dev']]
		tags += [ ('relative_darkness/%d' % narrow_size,  'relative_darkness_%d_%d' % (narrow_size, thresh)) for thresh in [10, 20]]
		tags += [(feature, '%s_%d' % (feature, wide_size)) for feature in ['mean', 'median']]
		tags += base_tags
		name = "channel_narrow_wide_%d" % narrow_size
		print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3})
		createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=True)

	for name, tags in TAG_SETS.items():
		name = "channel_%s_all" % name
		print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3})
		createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=True)


def channel3Experiments(ds):
	group = "channel3"
	base_tags = TAG_SETS["original"]

	tags = [base_tags[0],
		    'percentile', 'bilateral', 'canny',
			('relative_darkness/9', 'relative_darkness_9_10'), 
			('median', 'median_19'), 
			('median', 'median_9'),  
			('mean', 'mean_19'), 
			('min', 'min_9')
			]

	# combined narrow windows
	name = "channel_custom"
	print "createHisDbExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3})
	createNetwork.createHisDbExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=True)


if __name__ == "__main__":
	for ds in datasets:
		#howeFeaturesExperiments(ds)
		#oneConvExperiments(ds)
		#lrnExperiments(ds)
		#augmentExperiments(ds)
		#augment2Experiments(ds)
		#augment3Experiments(ds)
		#depthExperiments(ds)
		#scaleExperiments(ds)
		#scale2Experiments(ds)
		#scale3Experiments(ds)
		#scale4Experiments(ds)
		#widthExperiments(ds)
		#kernelSizeExperiments(ds)
		#featuresExperiments(ds)
		#features2Experiments(ds)
		#features3Experiments(ds)
		#channel2Experiments(ds)
		#channel3Experiments(ds)
		#lossExperiments(ds)
		#loss2Experiments(ds)
		#loss3Experiments(ds)
		#loss4Experiments(ds)
		#zeroExperiments(ds)
		#denseExperiments(ds)
		#poolExperiments(ds)
		#batchSizeExperiments(ds)
		#lrExperiments(ds)
		#arch512Experiments(ds)
		#finalExperiments(ds)
		arch2Experiments(ds)
		#arch3Experiments(ds)
		#residualExperiments(ds)
		#denseExperiments(ds)

