#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=6GB
#SBATCH -J "avg_results"
#SBATCH --gid=fslg_icdar
#SBATCH --mail-user=christopher.tensmeyer@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
echo "Cuda devices: $CUDA_VISIBLE_DEVICES"

script=/fslhome/waldol1/fsl_groups/fslg_icdar/compute/BYU-AWESOME/scripts/create_doc_datum_lmdbs.py

dir=/fslhome/waldol1/fsl_groups/fslg_icdar/compute/data/chris/hisdb

./avg_predictions2.sh ../../../experiments/hdibco_gray/nets/hdibco_gray/round_control/round_skip_2/ ../../../data/chris/hdibco_gray/ > /dev/null &
#./avg_predictions2.sh ../../../experiments/hdibco_gray/nets/hdibco_gray/round_control/round_5/ ../../../data/chris/hdibco_gray/ > /dev/null &
#./avg_predictions2.sh ../../../experiments/hdibco_gray/nets/hdibco_gray/round_control/round_9/ ../../../data/chris/hdibco_gray/ > /dev/null &
#./avg_predictions2.sh ../../../experiments/hdibco_gray/nets/hdibco_gray/round_control/round_skip_5/ ../../../data/chris/hdibco_gray/ > /dev/null &
#./avg_predictions2.sh ../../../experiments/hdibco_gray/nets/hdibco_gray/round_control/round_skip_9/ ../../../data/chris/hdibco_gray/ > /dev/null &

wait

