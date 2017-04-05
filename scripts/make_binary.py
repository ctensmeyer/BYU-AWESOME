
import os
import sys

out_dir = sys.argv[1]
try:
	os.makedirs(out_dir)
except:
	pass

prefix = sys.argv[2]
dataset = sys.argv[3]

for idx, f in enumerate(sys.argv[4:]):
	f = os.path.abspath(f.strip())
	out_f = os.path.join(out_dir, "%s_%d.sh" % (prefix, idx))
	
	with open(out_f, 'w') as out:
		out.write("#!/bin/bash\n\n")
		out.write("#SBATCH --time=168:00:00\n")
		out.write("#SBATCH --ntasks=6\n")
		out.write("#SBATCH --nodes=1\n")
		out.write("#SBATCH --gres=gpu:1\n")
		out.write("#SBATCH --mem-per-cpu=2666MB\n")
		out.write("#SBATCH -J \"%s-%s-%d\"\n" % (dataset, prefix, idx))
		out.write("#SBATCH --gid=fslg_nnml\n")
		out.write("#SBATCH --mail-user=christopher.tensmeyer@gmail.com\n")
		#out.write("#SBATCH --mail-type=BEGIN\n")
		out.write("#SBATCH --mail-type=END\n")
		out.write("#SBATCH --mail-type=FAIL\n")
		out.write("#SBATCH --qos=standby\n")
		out.write("#SBATCH --requeue\n\n")

		out.write("export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE\n")
		out.write("echo \"Cuda devices: $CUDA_VISIBLE_DEVICES\"\n\n")

		out.write("script=/fslhome/waldol1/fsl_groups/fslg_icdar/compute/experiments/binarize/jobs/binary.sh\n\n")
		out.write("dir=%s\n\n" % f)
		out.write("cp $script $dir\n")
		out.write("$dir/binary.sh 0 resume\n")

