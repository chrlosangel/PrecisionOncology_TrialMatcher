#!/bin/bash
#SBATCH --job-name=OncoMatch_patients
#SBATCH --output=../logs/OncoMatch_patients.log
#SBATCH --error=../logs/OncoMatch_patients.err
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=64G


source ~/.bashrc
conda activate trialmatch

patients=$1
db=$2
LLM_model=$3
embedding_model=$4
sentence_tokenizer=$5
# LLM here is not as important as the embedding model, so we can use a smaller model
python $(realpath "../src/preprocessing/run_patients.py") \
--patients_path ${patients} \
--LLM_model ${LLM_model} \
--embedding_model ${embedding_model} \
--save_dir ${db} \
--sentence_tokenizer ${sentence_tokenizer}

