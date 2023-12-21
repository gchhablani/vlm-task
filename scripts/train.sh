#!/bin/bash
#SBATCH --job-name=imagenav
#SBATCH --output=slurm_logs/imagenav-ddppo-%j.out
#SBATCH --error=slurm_logs/imagenav-ddppo-%j.err
#SBATCH --gpus a40:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@90
#SBATCH --requeue
#SBATCH --exclude=baymax,xaea-12,heistotron,gundam,consu
#SBATCH --partition=cvmlp-lab
#SBATCH --qos=long

export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

source /srv/flash1/gchhablani3/miniforge3/etc/profile.d/conda.sh
conda deactivate
conda activate vlm_task

TENSORBOARD_DIR="tb/imagenav/vc1_2/seed_1/"
CHECKPOINT_DIR="data/new_checkpoints/vc1_2/seed_1/"
DATA_PATH="data/datasets/pointnav_gibson_v1/"

JOB_ID="vc1_2"
WB_ENTITY="gchhablani"
PROJECT_NAME="vlm"

srun python -um vlm.run \
  --run-type train \
  --exp-config config/experiments/ddppo_imagenav_gibson.yaml \
  habitat_baselines.trainer_name="ddppo" \
  habitat_baselines.num_environments=16 \
  habitat.dataset.content_scenes=['Delton'] \
  habitat_baselines.num_updates=-1 \
  habitat_baselines.total_num_steps=1000000 \
  habitat_baselines.wb.entity=$WB_ENTITY \
  habitat_baselines.wb.run_name=$JOB_ID \
  habitat_baselines.wb.project_name=$PROJECT_NAME \
  habitat_baselines.writer_type=wb \
  habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
  habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
  habitat.dataset.data_path=${DATA_PATH}/train/train.json.gz \
  habitat.task.measurements.success.success_distance=0.25 \