#!/bin/bash
#SBATCH -c 10
#SBATCH -w ruapehu
#SBATCH --gres=gpu:1
#SBATCH --job-name=pistol
#SBATCH --tasks-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --time=10:00:00

#source $(poetry env info --path)/bin/activate

export HF_TOKEN=$(cat /nfs-share/mk2296/.huggingface_token)


#poetry run python eval.py

#conda activate pistol
which python
python eval.py forget.num_run=1 forget.lr=1e-5  