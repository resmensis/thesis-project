#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --account=wilms_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --output=/beegfs/uu/bbc9941/bbc9941/thesis-project/logs/slurm_%j.out
#SBATCH --export=NONE
#SBATCH --mail-user=simon.hennig@studium.uni-hamburg.de
#SBATCH --mail-type=BEGIN,END,FAIL

source /sw/batch/init.sh

set -e
set -o pipefail

module switch env env/2025Q1-gcc-openmpi
module load python

source "$USW/myvenv/bin/activate"

cd /beegfs/uu/bbc9941/bbc9941/thesis-project

mkdir -p logs output cache descriptives

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
command -v python
python --version

nvidia-smi

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Visible GPUs:', torch.cuda.device_count())"

python -u run_experiments.py
