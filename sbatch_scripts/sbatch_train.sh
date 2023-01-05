#!/bin/bash
#SBATCH -A research
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-0:00:00
#SBATCH --mail-user=amogh.tiwari@research.iiit.ac.in
#SBATCH --mail-type=ALL

#SBATCH --output=logs/sbatch_train_vanilla_log.txt

set -x

cd ~/personal_projects/food_c_challenge
bash initialize.sh
echo "initialized stuff"

conda init bash
eval "$(conda shell.bash hook)"
conda activate food_c # env_name
echo "Activated conda env"

# module add  cuda/9.0 cudnn/7.6.4-cuda-9.0
python train.py --model vanilla --lr 0.005 --dataroot data/food_c_data
