#!/bin/bash
#SBATCH --job-name=OncoMatch_interrogate
#SBATCH --output=../logs/OncoMatch_interrogate.log
#SBATCH --error=../logs/OncoMatch_interrogate.err
#SBATCH --time=40:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=64G


source ~/.bashrc
conda activate trialmatch

ct_processed=$1
patients_db=$2
LLM_model=$3
embedding_model=$4

python $(realpath ../src/matching/run_answering.py) \
--patients_database_path ${patients_db} \
--clinical_trials_file ${ct_processed} \
--embedding_model ${embedding_model} \
--LLM_model ${LLM_model}