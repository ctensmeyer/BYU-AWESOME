
#metric_script=$CAFFE_HOME/python/binary_eval.py
metric_script=./binary_eval.py

in_dir=${1%/}  # no trailing /
dir_1=$in_dir/1

dataset_dir=$2

if [ "$#" -ne 2 ]; then
	echo "args are: in_dir dataset_dir"
	exit 1
fi

echo $in_dir $dir_1 $dataset_dir

for split in train val test
do
	echo $split
    binary_dir1=$dir_1/predictions/$split/binary
	echo $binary_dir1
	for d in $binary_dir1/*
	do
		gt_name=`basename $d | rev | cut -d_ -f3- | rev`
		gt=`basename $d`
		echo $gt
		out_dir=${in_dir}_avg/1/predictions/$split/binary/$gt/
		results_dir=${in_dir}_avg/1/predictions/$split/results/
		xor_dir=${in_dir}_avg/1/predictions/$split/xor/$gt/
		mkdir -p $out_dir $results_dir $xor_dir

		im_dir=$in_dir/1/predictions/$split/binary/$gt/
		echo $im_dir

		for im in $im_dir/*
		do
			base=`basename $im`
			echo $base
			python avg_ims.py $out_dir/$base $in_dir/*/predictions/$split/binary/$gt/$base
		done
		python $metric_script $out_dir $dataset_dir/$gt_name $results_dir/${gt}.txt $results_dir/${gt}_summary.txt
	done
done
