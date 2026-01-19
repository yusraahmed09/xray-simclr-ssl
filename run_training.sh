#!/bin/bash
#
# SLURM DIRECTIVES: Configure the resources needed for your final job.
#
#SBATCH --job-name=SimCLR_Pretrain     # Name of job for the queue
#SBATCH --output=slurm_logs/slurm-%j.out  # Standard output log file (where prints go)
#SBATCH --error=slurm_logs/slurm-%j.err   # Standard error log file
#SBATCH --time=90:00:00                # Maximum job run time (3 days)
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --cpus-per-task=8              # Request 8 CPU cores (Matches your NUM_WORKERS=8 setting)
#SBATCH --gres=gpu:1                   # Request 1 GPU (the resource needed)

# --- SETUP ENVIRONMENT ---
echo "--- Starting job on node $(hostname) ---"

# Load the required modules
module load anaconda

# Activate your conda environment (crucial!)
conda activate simclr_env2

# Check GPU status (Optional, but good for logs)
nvidia-smi

cd /lustre/fs1/home/yu395012/SimCLR/

# --- EXECUTE PYTHON SCRIPT ---
echo "Running main training script: train.py"
# The Python script runs the full 72-hour loop
python train.py

echo "--- Job finished successfully ---"