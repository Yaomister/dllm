#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --array=0-11%4

cd $SLURM_SUBMIT_DIR

NUM_BATCHES=4
SEEDS=(0 1 2)
DATASET=${DATASET:?set DATASET=humaneval or mbpp}

idx=$SLURM_ARRAY_TASK_ID
batch=$(( idx % NUM_BATCHES ))
rest=$(( idx / NUM_BATCHES ))
d=$(( rest % 2 ))
s=$(( rest / 2 ))

mkdir -p logs

source /home/yao.eric/dllm-tpos-schedules/.venv/bin/activate

export HF_HOME=/scratch/yao.eric/hf_cache/huggingface


python -u main.py \
  --seed ${SEEDS[$s]} \
  --dataset ${DATASETS[$d]} \
  --batch $batch \
  --batch-size $NUM_BATCHES