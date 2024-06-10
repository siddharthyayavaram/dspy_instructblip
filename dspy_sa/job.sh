#!/bin/bash
#SBATCH -p gpu_a100_8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 32G
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH --job-name="full"
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --gres=gpu:1
#SBATCH --mail-user=f20213117@hyderabad.bits-pilani.ac.in
#SBATCH --mail-type=NONE
module load cuda-11.7.1-gcc-11.2.0-gypzm3r

# Get the Python file name from the command line argument
PYTHON_FILE=$1

# Run the specified Python file
srun python $PYTHON_FILE