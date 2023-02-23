#!/bin/bash
#
#SBATCH --job-name=train # Job name for tracking
#SBATCH --partition=falcon # Partition you wish to use (see above for list)
#SBATCH --cpus-per-task=6      # Number of CPU cores used by your job
#SBATCH --mem=60000           # RAM in MB needed in total (also available --mem-per-cpu)
#SBATCH --time=48:00:00        # Job limited to 12 hours (see sinfo for maximums)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=./output.out # Standard out from your job
#SBATCH --error=./error.err  # Standard error from your job

## Execute your program(s) ##
. /etc/profile.d/modules.sh
module load CUDA

srun --nodelist="falcon-04" python3 -m pipenv run python ./train/train.py configs/asap.yaml