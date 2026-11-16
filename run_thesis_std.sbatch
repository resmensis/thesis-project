#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=std
#SBATCH --account=wilms_std
#SBATCH --time=12:00:00
#SBATCH --output=/beegfs/uu/bbc9941/bbc9941/thesis-project/logs/slurm_%j.out
#SBATCH --mail-user=simon.hennig@studium.uni-hamburg.de
#SBATCH --mail-type=BEGIN,END,FAIL

source /sw/batch/init.sh

module switch env env/2025Q1-gcc-openmpi

source $USW/myvenv/bin/activate

cd /beegfs/uu/bbc9941/bbc9941/thesis-project

mkdir -p logs output cache descriptives

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

python run_experiments.py
