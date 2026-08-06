#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out

cd $SLURM_SUBMIT_DIR

mkdir -p training logs

source /home/yao.eric/dllm-tpos-schedules/.venv/bin/activate

python main.py --experiment "$1"