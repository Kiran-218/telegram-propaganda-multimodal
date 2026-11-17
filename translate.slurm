#!/bin/bash
#SBATCH --job-name=nllb-translate
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/translate_%j.out
#SBATCH --error=logs/translate_%j.err

# --- environment ------------------------------------------------------------
# Source conda explicitly so the job works on any node, regardless of
# whether the login shell initialised conda.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlp

# Keep HF fully offline-friendly and cache models in the project dir so the
# 1.3B weights download once and are reused on later runs.
export HF_HOME="$HOME/dissertation/translation/.hf_cache"

# --- run --------------------------------------------------------------------
cd ~/dissertation/translation

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python translate_captions_gpu.py \
    --in  data/annotation_sample.csv \
    --out output/annotation_sample_translated_1.3B.csv

echo "Finished at $(date)"
