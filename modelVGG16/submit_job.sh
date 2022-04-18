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
#pip3 install opencv-python==4.5.5.62 --user
module load TensorFlow/2.5.0-fosscuda-2020b
#module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4
#module load OpenCV/4.2.0-fosscuda-2019b-Python-3.7.4
#module load SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4
#module load matplotlib/3.1.1-foss-2019b-Python-3.7.4
# module load Seaborn/0.11.2-foss-2021a

# pip3 install -r requirements.txt --user
# pip3 install tensorflow-addons --user
pip3 install scikit-plot --user
pip3 install seaborn --user
pip3 install imbalanced-learn --user
pip3 install scipy --user
pip3 install six numpy scipy Pillow matplotlib scikit-image opencv-python-headless imageio
pip3 install --no-dependencies imgaug

# Run instruction

# Run instruction
python3 VGGImgAugNewData.py