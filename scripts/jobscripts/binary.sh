#!/bin/bash

gpu=$1
resume=$2

if [[ -z $gpu ]]; then
	echo "No GPU specified"
	exit 1
fi

# training
net_dir=`dirname $0`
snapshot_dir=$net_dir/snapshots
log_dir=$net_dir/logs
test_log_dir=$net_dir/test_logs
graph_dir=$net_dir/graphs
train_model_file=$net_dir/train_train.prototxt
val_solver_file=$net_dir/solver.prototxt
val_model_file=$net_dir/train_val.prototxt
test_model_file=$net_dir/train_test.prototxt
result_file=$net_dir/results.txt
train_log_file=$log_dir/caffe_$$.log.INFO
deploy_file=$net_dir/deploy.prototxt

train_test_log_file=$test_log_dir/caffe_$$_train.log.INFO
val_test_log_file=$test_log_dir/caffe_$$_val.log.INFO
test_test_log_file=$test_log_dir/caffe_$$_test.log.INFO

# TODO: change this to handle all the different outputs
# TODO: Adative tile size for prediction
# predict
best_model_script=/fslhome/waldol1/fsl_groups/fslg_icdar/compute/BYU-AWESOME/scripts/get_best_model.py
prediction_script=/fslhome/waldol1/fsl_groups/fslg_icdar/compute/BYU-AWESOME/scripts/binary_prediction.py
eval_script=/fslhome/waldol1/fsl_groups/fslg_icdar/compute/BYU-AWESOME/scripts/binary_eval.py
xor_script=/fslhome/waldol1/fsl_groups/fslg_icdar/compute/BYU-AWESOME/scripts/binary_image_diff.py
out_dir=$net_dir/predictions
dataset=`readlink -f $net_dir | rev | cut -d\/ -f4 | rev`
dataset_dir=/fslhome/waldol1/fsl_groups/fslg_icdar/compute/data/chris/$dataset
labels_dir=$dataset_dir/labels

echo "Training network in $net_dir on GPU $gpu of Host `hostname`"
echo "Solver File: $val_solver_file"
echo "Model File: $val_model_file"

mkdir -p $snapshot_dir
mkdir -p $log_dir
mkdir -p $test_log_dir
mkdir -p $graph_dir

echo "Location of CAFFE_HOME: $CAFFE_HOME"

if [[ ! -z $resume ]] && [[ $(ls -A $snapshot_dir) ]]; then
	# train from the latest snapshot
	solverstate_file=`find $snapshot_dir -name "*.solverstate" -printf "%T+ %p\n" | sort -r | head -n1 | cut -f2 --delimiter=' '`
	modelstate_file=`find $snapshot_dir -name "*.caffemodel" -printf "%T+ %p\n" | sort -r | head -n1 | cut -f2 --delimiter=' '`

	echo "Resuming training from solverstate: $solverstate_file and model weights: $modelstate_file"
	ln -sf $train_log_file $log_dir/caffe.INFO
	$CAFFE_HOME/build/tools/caffe train -solver $val_solver_file -snapshot $solverstate_file --gpu $gpu 2>&1 | tee $train_log_file
else
	echo "Training from scratch"
	echo "Cleaning up past models"
	rm -r $snapshot_dir $log_dir $graph_dir $test_log_dir
	mkdir -p $snapshot_dir $log_dir $graph_dir $test_log_dir
	rm  $result_file

	ln -s $train_log_file $log_dir/caffe.INFO
	$CAFFE_HOME/build/tools/caffe train -solver $val_solver_file --gpu $gpu 2>&1 | tee $train_log_file
fi

echo "Done Training"

# graph log file(s)
echo "Concatenating log files and graphing results"
combined_log=$log_dir/combined.txt
logs=`find $log_dir -name "*log.INFO*" -printf "%T+ %p\n" | sort | cut -f2- --delimiter=' '`
echo "LOGS: $logs"
cat $logs > $combined_log
python $CAFFE_HOME/python/graph_log.py $combined_log $graph_dir

# find best validation network
best_model_num=`python $best_model_script $combined_log`
prefix=`find $snapshot_dir -type f | head -n1`
prefix=`basename $prefix | rev | cut -d_ -f2- | rev`
echo $prefix
best_model=$snapshot_dir/${prefix}_${best_model_num}.caffemodel
echo "Best model: $best_model"
if [[ ! -e $best_model ]]; then
  echo "Best Model does not exist"
  exit 1
fi
cp $best_model $net_dir/best_model_`basename $best_model`
cp $best_model $net_dir/best_model.caffemodel

# test best validation network on train set
echo "Testing best network on training data"
lmdb=`grep "sources:" $train_model_file | tail -n1 | cut --delimiter=' ' -f6 | cut --delimiter='"' -f2`
num_entries=`mdb_stat $lmdb | grep Entries | cut -d' ' -f4`
echo "Number of entries: $num_entries"
$CAFFE_HOME/build/tools/caffe test -model $train_model_file -weights $best_model --gpu $gpu -iterations $num_entries 2>&1 | tee $train_test_log_file
echo > $result_file
echo "Train: " >> $result_file
#tail -n6 $train_test_log_file | cut --delimiter=' ' -f5-  | cut --delimiter=] -f2 | sed 's/^ *//' | cut --delimiter='(' -f1 >> $result_file
grep 'loss)$' $train_test_log_file | cut --delimiter=' ' -f5-  | cut --delimiter=] -f2 | sed 's/^ *//' | cut --delimiter='(' -f1 >> $result_file


# test best validation network on validation set
echo "Testing best network on validation data"
lmdb=`grep "sources:" $val_model_file | tail -n1 | cut --delimiter=' ' -f6 | cut --delimiter='"' -f2`
num_entries=`mdb_stat $lmdb | grep Entries | cut -d' ' -f4`
echo "Number of entries: $num_entries"
$CAFFE_HOME/build/tools/caffe test -model $val_model_file -weights $best_model --gpu $gpu -iterations $num_entries 2>&1 | tee $val_test_log_file
echo >> $result_file
echo "Validation: " >> $result_file
#tail -n6 $val_test_log_file | cut --delimiter=' ' -f5-  | cut --delimiter=] -f2 | sed 's/^ *//' | cut --delimiter='(' -f1 >> $result_file
grep 'loss)$' $val_test_log_file | cut --delimiter=' ' -f5-  | cut --delimiter=] -f2 | sed 's/^ *//' | cut --delimiter='(' -f1 >> $result_file
echo "num_iters = $best_model_num" >> $result_file


# test best validation network on test set
echo "Testing best network on test data"
lmdb=`grep "sources:" $test_model_file | tail -n1 | cut --delimiter=' ' -f6 | cut --delimiter='"' -f2`
num_entries=`mdb_stat $lmdb | grep Entries | cut -d' ' -f4`
echo "Number of entries: $num_entries"
$CAFFE_HOME/build/tools/caffe test -model $test_model_file -weights $best_model --gpu $gpu -iterations $num_entries 2>&1 | tee $test_test_log_file
echo >> $result_file
echo "Test: " >> $result_file
#tail -n6 $test_test_log_file | cut --delimiter=' ' -f5-  | cut --delimiter=] -f2 | sed 's/^ *//' | cut --delimiter='(' -f1 >> $result_file
grep 'loss)$' $test_test_log_file | cut --delimiter=' ' -f5-  | cut --delimiter=] -f2 | sed 's/^ *//' | cut --delimiter='(' -f1 >> $result_file


sources=`cat $net_dir/inputs.txt | paste -sd,`
tile_size=`grep input_dim: $deploy_file | tail -n1 | cut -d' ' -f2`
#tile_size=512
pad_size=$(( $tile_size / 4 ))

echo "Predicting Images"
echo "Dataset dir: $dataset_dir"

for split in train val test
do
	python $prediction_script -t $tile_size -p $pad_size --im-dirs $sources --gpu $gpu \
		$deploy_file $best_model $dataset_dir $labels_dir/${split}.txt $out_dir/$split 

	mkdir -p $out_dir/$split/results

	# run binary evaluation over each set of output
	for d in $out_dir/$split/binary/*
	do
		gt=`basename $d | rev | cut -d_ -f3- | rev`
		base=`basename $d`
		python $eval_script $d $dataset_dir/$gt $out_dir/$split/results/${base}.txt $out_dir/$split/results/${base}_summary.txt

		mkdir -p $out_dir/$split/xor/$base
		for f in $d/*
		do
			f_base=`basename $f`
			python $xor_script $f $dataset_dir/$gt/$f_base $out_dir/$split/xor/$base/$f_base
		done
	done
done


# clean up snapshots
echo "Cleaning up snapshots"
touch $best_model
ls -tp $snapshot_dir/*.solverstate | grep -v '/$' | tail -n +6 | xargs -I {} rm -- {}
ls -tp $snapshot_dir/*.caffemodel | grep -v '/$' | tail -n +6 | xargs -I {} rm -- {}

echo "Done"
