#!/bin/bash
# Second Tello training. The first one (slurm/train.slurm) held the target at about 20 cm against 4 cm for the
# driver's hold: one network update per 4 transitions (125k in 500k steps) and a steady offset of 5-20 cm. Here:
#   tasks 0-4: one update per transition (--gradient-steps 4), 500k steps
#   tasks 5-9: the same with the closeness bonus exp(-(d / 0.15 m)^2) of the airplane (--bonus 1)
# About 3 h per task on one core. Submit from the repository root:  sbatch slurm/train_v2.slurm
#SBATCH --job-name=tello-rl-v2
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
if [ "$SLURM_ARRAY_TASK_ID" -lt 5 ]; then NAME=gs4; EXTRA=""; else NAME=gs4_bonus; EXTRA="--bonus 1"; fi
python train.py --steps 500000 --seed "$SEED" --n-envs 4 --gradient-steps 4 $EXTRA \
    --out "runs/${NAME}_${SLURM_ARRAY_JOB_ID}_seed${SEED}"
