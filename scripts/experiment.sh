#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out

cd $SLURM_SUBMIT_DIR

NUM_BATCHES = 4
SEEDS=(0 1 2)
DATASETS=(math gsm8k)

idx=$SLURM_ARRAY_TASK_ID
batch=$(( idx % NUM_BATCHES ))
rest=$(( idx / NUM_BATCHES ))
d=$(( rest % 2 ))
s=$(( rest / 2 ))




mkdir -p training logs

source /home/yao.eric/dllm-tpos-schedules/.venv/bin/activate

python -u main.py \
  --seed ${SEEDS[$s]} \
  --dataset ${DATASETS[$d]} \
  --batch $batch \
  --batch-size $NUM_BATCHES