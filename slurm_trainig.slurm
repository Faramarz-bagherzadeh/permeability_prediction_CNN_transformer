#!/bin/bash -l
#SBATCH --account=glaz.glaz
#SBATCH --partition=gpu
#SBATCH --gpus=a100:4
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --qos=48h
#SBATCH --cpus-per-task=120
#SBATCH --time=47:00:00
#SBATCH --job-name=Train
#SBATCH --output=logs/train_%A_%a.out

PROJECT_DIR="${1:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_DIR" ]]; then
    echo "Repository directory argument or SLURM_SUBMIT_DIR is required" >&2
    exit 1
fi
CONFIG_PATH="${2:-$PROJECT_DIR/config.yaml}"
cd "$PROJECT_DIR"

module load mesa
module load cuda
module load conda
conda activate perm-pred-env

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    python -u train.py --config "$CONFIG_PATH" --model-index "$SLURM_ARRAY_TASK_ID"
else
    python -u train.py --config "$CONFIG_PATH"
fi