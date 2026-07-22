#!/bin/bash
#SBATCH --job-name=OncoMatch_trials
#SBATCH --output=../logs/OncoMatch_trials.log
#SBATCH --error=../logs/OncoMatch_trials.err
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=64G


source ~/.bashrc
conda activate trialmatch

ct=$1
savedir=$2
LLM_model=$3
embedding_model=$4
#LLM_model="meta-llama/Llama-3.3-70B-Instruct"

python $(realpath ../src/preprocessing/run_trials.py) \
--clinical_trials_path ${ct} \
--cancer_filter "False" \
--LLM_model ${LLM_model} \
--embedding_model ${embedding_model} \
--save_dir ${savedir}
