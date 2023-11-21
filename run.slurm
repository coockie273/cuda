#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH -p tornado-k40
#SBATCH -t 10-00:00:00
#SBATCH -J Markup
#SBATCH -o Markup-%j.out
#SBATCH -e Markup-%j.err

if [ -f etc/profile.d/modules-basis.sh ]; then
   source /etc/profile.d/modules-basis.sh
fi

module purge
module load compiler/gcc/6
module load nvidia/cuda/9.2
nvcc markup.cu -o markup
./markup 1 2 3