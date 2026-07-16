#!/bin/bash
#SBATCH --job-name=trials
#SBATCH --output=logs/trials.log
#SBATCH --error=logs/trials.err
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=42G


source ~/.bashrc
conda activate trialmatch

ct=$(realpath "./data/synthetic/trials/coral_new/")
savedir=$(realpath "./database/chromadb/")

python src/preprocessing/run_trials.py \
--clinical_trials_path ${ct} \
--cancer_filter "False" \
--LLM_model "Qwen/Qwen1.5-14B-Chat" \
--embedding_model "ncbi/MedCPT-Query-Encoder" \
--save_dir ${savedir}
