#!/bin/bash
#SBATCH --job-name=cache_vc1
#SBATCH --output=slurm_logs/dataset-%j.out
#SBATCH --error=slurm_logs/dataset-%j.err
#SBATCH --gpus 2080_ti:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --exclude=conroy,ig-88
#SBATCH --signal=USR1@100
#SBATCH --requeue
#SBATCH --partition=cvmlp-lab
#SBATCH --qos=short

export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

source /srv/flash1/gchhablani3/miniforge3/etc/profile.d/conda.sh
conda deactivate
conda activate vlm_task

srun python vlm/utils/cache_image_goals.py \
  --split train \
  --config config/experiments/ddppo_imagenav_gibson.yaml \
  --input-path data/datasets/pointnav_gibson_v1 \
  --output-path data/datasets/vc1_embeddings \
  --scene Delton
