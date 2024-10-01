#!/bin/bash
#SBATCH --job-name=exp01           # Job name
#SBATCH --nodes=1        
##SBATCH --nodelist=crannog05                # Number of nodes
#SBATCH --gres=gpu:1             # Number of GPUs required
##SBATCH --partition=PGR-Standard
#SBATCH --time=3-00:00:00              # Walltime
##SBATCH --output=logs/%A_%a.out
##SBATCH --error=logs/%A_%a.err
#SBATCH --array=1-4  # Specifies the range of tasks in the array

source /opt/conda/bin/activate
conda activate moad
python render_abc.py
