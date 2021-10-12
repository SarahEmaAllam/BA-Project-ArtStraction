#!/bin/bash
# Information about job
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=12G
#SBATCH --mail-type=END
#SBATCH --mail-user=s.allam@student.rug.nl
#SBATCH --partition=gpushort

module purge
module load

srun mdrun