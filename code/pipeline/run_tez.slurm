#!/bin/bash

#SBATCH -J v10_ultimate

#SBATCH -p cpu2dq

#SBATCH -A dtihtv 

#SBATCH -N 1

#SBATCH -n 1

#SBATCH -c 20

#SBATCH --time=47:59:00

#SBATCH --mem=64G

#SBATCH --output=ultimate-%j.out

#SBATCH --error=ultimate-%j.err

# --- MAIL AYARLARI (EKLENDI) ---
#SBATCH --mail-type=ALL          # BEGIN, END, FAIL durumlarinda mail atar
#SBATCH --mail-user=boyabatli23@itu.edu.tr



echo "=================================================================="

echo "   Code SERVER RUN - START: $(date)"

echo "=================================================================="





# Klasöre git

cd /okyanus/users/kboyabatli/yl_tez_calismasi



# Modülleri yükle

module load Anaconda/Anaconda3-2022.05-python3.9.12

source /okyanus/progs/ANACONDA/Anaconda3-2022.05-python3.9/etc/profile.d/conda.sh



/okyanus/users/kboyabatli/.conda/envs/tez_env/bin/python -u uhem_big_optuna_v13.py



echo "=================================================================="

echo "   ISLEM TAMAMLANDI: $(date)"

echo "=================================================================="