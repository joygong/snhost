#!/bin/bash -l
#SBATCH -t 12:00:00
#SBATCH --job-name=ZTF_SED_FIT
#SBATCH --account=PEIRIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jjg65@cam.ac.uk

# the directives above are setting:
# -t          | wall time
# --job-name  | job name to appear in the queue
# --account   | which account to bill the job to
# --partition | ampere = GPU nodes
# --nodes     | number of nodes
# --gres      | number of GPU cards (max 4 per node)
# --mail-type | when to send you an email
# --mail-user | your email address

# load anything necessary
# your .bashrc should get run anyway
module purge
module load cuda/11.8
module load anaconda
conda activate myenv

# run script
python MY_SCRIPT.py
