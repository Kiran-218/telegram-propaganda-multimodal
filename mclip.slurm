#!/bin/bash
#SBATCH --job-name=mclip-score
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mclip_%j.out
#SBATCH --error=logs/mclip_%j.err

# Source conda so the job works on any compute node.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mclip

# Cache M-CLIP / CLIP weights in the project dir (already pre-downloaded on the
# head node, so the job loads from cache).
export HF_HOME="$HOME/dissertation/mclip/.hf_cache"

cd ~/dissertation/mclip

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --media-root points at the folder that CONTAINS media/ ; adjust if the images
# live elsewhere on the cluster.
python mclip_score.py \
    --in  data/annotation_sample.csv \
    --out output/mclip_scores.csv \
    --media-root .

echo "Finished at $(date)"
