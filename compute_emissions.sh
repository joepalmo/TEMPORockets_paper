#!/bin/bash
#SBATCH -J rocket_emissions
#SBATCH -p edr
#SBATCH -t 1-00:00
#SBATCH -N 1 
#SBATCH -c 30
#SBATCH --mem=40000
#send email when job ends or fails
#SBATCH --mail-user=jpalmo@mit.edu
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

# Load the necessary modules
source /etc/profile.d/modules.sh
module load anaconda3
source activate earthdata

# run the Python script
python compute_emissions.py
