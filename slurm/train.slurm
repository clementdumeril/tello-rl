#!/bin/bash
# SAC on the identified Tello model (tello_env.py), 5 seeds, 500k steps each (about 1 h on one core), then the
# tests of evaluate.py against the driver's hold controller. Uses the Python of the Fixed-Wing venv on Savio
# (torch CPU, stable-baselines3, gymnasium). Submit from the repository root:
# sbatch slurm/train.slurm
#SBATCH --job-name=tello-rl
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --array=0-4
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python train.py --steps 500000 --seed "$SLURM_ARRAY_TASK_ID" --n-envs 4 \
    --out "runs/sac_${SLURM_ARRAY_JOB_ID}_seed${SLURM_ARRAY_TASK_ID}"
