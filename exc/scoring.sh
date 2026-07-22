#!/bin/bash
#SBATCH --job-name=OncoMatch_scoring
#SBATCH --output=../logs/OncoMatch_scoring.log
#SBATCH --error=../logs/OncoMatch_scoring.err
#SBATCH --time=20:00:00
#SBATCH --partition=cpu
#SBATCH --mem=30G

source ~/.bashrc
conda activate trialmatch

patients_database_path=$1
clinical_trials_file=$2

python $(realpath ../src/matching/scoring.py) \
    --patients_database_path "$patients_database_path" \
    --clinical_trials_file "$clinical_trials_file" \
    --run_all_patients