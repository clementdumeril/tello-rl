#!/bin/bash
# The Tello following circles and figure-eights (tello_track_env.py), 500k steps, one network update per transition
# (--gradient-steps 4), then the tests of evaluate_track.py against our model-based tracker:
#   tasks 0-4: with the closeness bonus exp(-(d / 0.15 m)^2) (--bonus 1)
#   tasks 5-9: without it
# About 3 h per task on one core.
# Submit from the repository root:  sbatch slurm/train_track.slurm
#SBATCH --job-name=tello-rl-track
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --array=0-9
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
SEED=$((SLURM_ARRAY_TASK_ID % 5))
if [ "$SLURM_ARRAY_TASK_ID" -lt 5 ]; then NAME=track_bonus; EXTRA="--bonus 1"; else NAME=track; EXTRA=""; fi
python train.py --task track --steps 500000 --seed "$SEED" --n-envs 4 --gradient-steps 4 $EXTRA \
    --out "runs/${NAME}_${SLURM_ARRAY_JOB_ID}_seed${SEED}"
