#!/bin/bash
#SBATCH --job-name=interrogate
#SBATCH --output=logs/interrogate.log
#SBATCH --error=logs/interrogate.err
#SBATCH --time=40:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=42G


source ~/.bashrc
conda activate trialmatch

ct_processed=$(realpath "./database/chromadb/processed_trials.pkl")
patients_db=$(realpath "./database/chromadb/chromaDB_patients")

python src/matching/run_matching.py \
--patients_database_path ${patients_db} \
--clinical_trials_file ${ct_processed} \
--embedding_model "ncbi/MedCPT-Query-Encoder" 
