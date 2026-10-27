#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --job-name=gp_dist
#SBATCH --account=YOURPROJECT-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/jjg65/rds/hpc-work/logs/gp_dist_%j.out
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
module purge
module load cuda/11.8
module load anaconda
conda activate myenv

# run script
cd /home/jjg65

# NOTE: RUN_NAME is deliberately NOT timestamped. The script checkpoints into
# $OUT/chunks and skips chunks already on disk, so a requeued or extended job
# resumes where it stopped. A new timestamp per submission would restart from zero.
RUN_NAME=cold_K5120_e2000

# --- fit settings ---
K=5120                # realizations to fit; leave empty for all available
CHUNK=32              # realizations per batched fit; raise if GPU memory allows
EPOCHS=2000
LR=0.05
GAMMA=0.998           # lr decay per step: 0.05 -> ~9e-4 over 2000 epochs

# --- precision / device ---
DTYPE=float64         # float64 is cheap on A100 and better conditioned for Cholesky
DEVICE=cuda
MERGE_DEVICE=cuda     # set to cpu if DIAG_FLAG is empty (full covariances are ~16 GB)

# --- initialization: empty INIT_FLAG = cold start ---
# INIT_FLAG="--init /home/jjg65/rds/hpc-work/gp-4/base_optimal_params_og.pth"
INIT_FLAG=""

# --- store variances only; drop this to keep full (K, n, n) covariances ---
DIAG_FLAG="--diag-only"
# DIAG_FLAG=""

# tripp direcotry for bias calculations
TRIPP_DIR=/home/jjg65/rds/hpc-work/ztf_sn

OUT=/home/jjg65/rds/hpc-work/gp-4-dist/${RUN_NAME}
mkdir -p $OUT
# mkdir -p /home/jjg65/rds/hpc-work/logs

# nvidia-smi

python script/run_gp_dist.py \
    --out $OUT \
    --K $K \
    --chunk $CHUNK \
    --epochs $EPOCHS \
    --lr $LR \
    --gamma $GAMMA \
    --dtype $DTYPE \
    --device $DEVICE \
    --merge-device $MERGE_DEVICE \
    --tripp-dir $TRIPP_DIR \
    $DIAG_FLAG \
    $INIT_FLAG