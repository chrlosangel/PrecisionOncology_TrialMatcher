#!/bin/bash
#SBATCH --job-name=patients
#SBATCH --output=logs/patients.log
#SBATCH --error=logs/patients.err
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=42G


source ~/.bashrc
conda activate trialmatch

patients=$(realpath "./data/coral/toy_set/")
db=$(realpath "./database/chromadb/")

python src/preprocessing/run_patients.py \
--patients_path ${patients} \
--LLM_model "Qwen/Qwen1.5-14B-Chat" \
--embedding_model "ncbi/MedCPT-Query-Encoder" \
--save_dir ${db} \
--sentence_tokenizer "en_core_sci_sm"

