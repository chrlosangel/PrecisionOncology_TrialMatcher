#!/bin/bash
#SBATCH --job-name=interrogate
#SBATCH --output=logs/interrogate.log
#SBATCH --error=logs/interrogate.err
#SBATCH --time=40:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=64G


source ~/.bashrc
conda activate trialmatch

ct_processed=$(realpath "./database/chromadb/processed_trials.pkl")
patients_db=$(realpath "./database/chromadb/chromaDB_patients")
LLM_model="Qwen/Qwen2.5-32B-Instruct"

python src/matching/run_answering.py \
--patients_database_path ${patients_db} \
--clinical_trials_file ${ct_processed} \
--embedding_model "ncbi/MedCPT-Query-Encoder" \
--LLM_model ${LLM_model}