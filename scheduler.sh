#!/bin/bash

#Absolute path to the project directory. Please modify as needed
PROJECT_DIR="/users/jcc2340/g2lab/projects/precision_oncology/"

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate trialmatch

bash ${PROJECT_DIR}/main.sh \
     --patients_path ${PROJECT_DIR}/data/coral/toy_set/" \
     --db_path ${PROJECT_DIR}/database/chromadb/" \
     --clinical_trials_path ${PROJECT_DIR}/data/synthetic/trials/coral_new/" \
     --LLM_model "Qwen/Qwen2.5-32B-Instruct" \
     --embedding_model "ncbi/MedCPT-Query-Encoder"
