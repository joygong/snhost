#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --job-name=SDE_DIST
#SBATCH --account=PEIRIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/jjg65/slurm_outputs/%x_%j.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jjg65@cam.ac.uk

# the directives above are setting:
# -t          | wall time
# --job-name  | job name to appear in the queue
# --account   | which account to bill the job to
# --partition | ampere = GPU nodes
# --nodes     | number of nodes
# --gres      | number of GPU cards (max 4 per node)
# --output    | stdout/stderr log (rds, not home -- quota)
# --mail-type | when to send you an email
# --mail-user | your email address

# load anything necessary
# your .bashrc should get run anyway
set -e          # stop at the first failure instead of limping on

module purge
module load cuda/11.8
module load anaconda

source $(conda info --base)/etc/profile.d/conda.sh
conda activate myenv

export MPLBACKEND=Agg          # diffusion.py imports pyplot; no display on compute nodes

python -c "import torch, flowfusion; print(torch.__version__, torch.cuda.is_available())"

# run script
cd /home/jjg65

# NOTE: RUN_NAME is deliberately NOT timestamped. The script checkpoints into
# $OUT/chunks and skips chunks already on disk, so a requeued or extended job
# resumes where it stopped. A new timestamp per submission would restart from zero.
RUN_NAME=K5120_N100

# --- fit settings ---
K=5120                # realizations; one model trained per realization
CHUNK=8               # realizations per checkpoint file (smaller = more frequent saves)
EPOCHS=200            # NOTE: diffusion.train runs 4 (bs, lr) stages, so this is 800 passes
N_COND=6              # x1, c, M, Z, sSFR, dust2
SEED=0                # same seed every realization, so training noise is common-mode

# --- sampling ---
N_SAMPLES=100         # draws per SN per realization
STEPS=100             # denoising steps
BATCH_SIZE=64         # SNe per sample_sde call; N_SAMPLES x BATCH_SIZE rows at once

# --- precision / device ---
DTYPE=float32
DEVICE=cuda

# tripp directory for bias calculations
TRIPP_DIR=/home/jjg65/rds/hpc-work/ztf_sn

OUT=/home/jjg65/rds/hpc-work/sde-4-dist/${RUN_NAME}
mkdir -p $OUT
mkdir -p /home/jjg65/rds/hpc-work/slurm_outputs

nvidia-smi

python script/run_diff_dist.py \
    --out $OUT \
    --K $K \
    --chunk $CHUNK \
    --n-samples $N_SAMPLES \
    --steps $STEPS \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --n-cond $N_COND \
    --seed $SEED \
    --dtype $DTYPE \
    --device $DEVICE \
    --tripp-dir $TRIPP_DIR