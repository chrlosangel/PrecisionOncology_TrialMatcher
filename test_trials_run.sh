#!/bin/bash
#SBATCH --job-name=trials
#SBATCH --output=logs/trials.log
#SBATCH --error=logs/trials.err
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=64G


source ~/.bashrc
conda activate trialmatch

ct=$(realpath "./data/synthetic/trials/coral_new/")
savedir=$(realpath "./database/chromadb/")
LLM_model="Qwen/Qwen2.5-32B-Instruct"
#LLM_model="meta-llama/Llama-3.3-70B-Instruct"

python src/preprocessing/run_trials.py \
--clinical_trials_path ${ct} \
--cancer_filter "False" \
--LLM_model ${LLM_model} \
--embedding_model "ncbi/MedCPT-Query-Encoder" \
--save_dir ${savedir}
