#!/bin/bash
#SBATCH --job-name=pneumonia_train
#SBATCH --output=logs_training_%j.out
#SBATCH --error=logs_training_%j.err
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=02:00:00

echo "Starting Slurm job execution..."
echo "Running on node: $SLURM_NODENAME"

# Force activate using explicit Python 3.12 tracking rules
export PATH="/home/hardy/ml_env_sandbox/linux_tf_env/bin:$PATH"
source /home/hardy/ml_env_sandbox/linux_tf_env/bin/activate

# Execute using the explicit version lock
python3.12 /mnt/d/Programming-large-dataset/ML-imaging-pneumonia/CNN-4.py

echo "Slurm job execution finalized."
