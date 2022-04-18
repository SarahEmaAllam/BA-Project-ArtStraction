#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:50:00
#SBATCH --mem=32G
#SBATCH --mail-type=END
#SBATCH --mail-user=s.allam@student.rug.nl
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:v100:1

# Load modules
module purge
pip3 install --upgrade pip --user
module load TensorFlow/2.5.0-fosscuda-2020b

pip3 install scikit-plot --user
pip3 install seaborn --user
pip3 install imbalanced-learn --user
pip3 install scipy --user
pip3 install six numpy scipy Pillow matplotlib scikit-image opencv-python-headless imageio
pip3 install --no-dependencies imgaug


# Run instruction
python3 ResNetFreeze.py