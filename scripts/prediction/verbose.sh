
#metric_script=$CAFFE_HOME/python/binary_eval.py
metric_script=./binary_eval.py

in_dir=$1 
dataset_dir=$2
original_image_dir=$dataset_dir/original_images

if [ "$#" -ne 2 ]; then
	echo "args are: in_dir dataset_dir"
	exit 1
fi

echo $in_dir $dataset_dir

for split in train val test
do
	echo $split
    verbose_dir=$in_dir/predictions/$split/verbose
	mkdir -p $verbose_dir

    binary_dir=$in_dir/predictions/$split/binary
	xor_dir=$in_dir/predictions/$split/xor
	raw_dir=$in_dir/predictions/$split/raw

	for d in $binary_dir/*
	do
		gt_name=`basename $d | rev | cut -d_ -f3- | rev`
		gt_dir=$dataset_dir/$gt_name
		gt=`basename $d`
		echo $gt

		for im in $binary_dir/$gt/*
		do
			base=`basename $im`
			echo $base
			out_dir=$verbose_dir/${base%.png}
			mkdir -p $out_dir

			cp $original_image_dir/$base $out_dir/original_im.png
			cp $gt_dir/$base $out_dir/gt.png
			python mult.py $out_dir/gt.png
			cp $raw_dir/$gt/$base $out_dir/raw.png
			cp $binary_dir/$gt/$base $out_dir/pred.png
			cp $xor_dir/$gt/$base $out_dir/xor.png
			python blend_ims.py $out_dir/xor.png $out_dir/original_im.png $out_dir/xor_on_original.png
			python blend_ims.py $out_dir/gt.png $out_dir/original_im.png $out_dir/gt_on_original.png 'invert'
			python blend_ims.py $out_dir/pred.png $out_dir/original_im.png $out_dir/pred_on_original.png 'invert'
		done
	done
done
