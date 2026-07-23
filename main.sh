#!/bin/bash
#SBATCH --job-name=OncoMatch
#SBATCH --output=./OncoMatch_%j.log
#SBATCH --error=./OncoMatch_%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=cpu
#SBATCH --mem=30G

#Needs to be send every night automatically to the cluster. This script will submit the jobs to the cluster and run the pipeline.

if command -v conda &> /dev/null; then
    source ~/.bashrc
    eval "$(conda shell.bash hook)"  # Initialize Conda properly
    conda activate trialmatch
    echo "[INFO] Using conda environment trialmatch. Python 3.11.15 "

else
    echo "[ERROR] Conda is not installed or not in PATH. Please install Conda."
    exit 1
fi

# Add parameters for user to input

usage() {
    echo ""
    echo "Usage: $0 [options]"
    echo "--patients_path <path>      Path to the patients data directory (required=True)"
    echo "--db_path <path>         Path to store the patients database (required=False, default=./database/chromadb/)"
    echo "--clinical_trials_path <path> Path to the clinical trials data directory (required=False)"
    echo "--LLM_model <model>          LLM model to use (required=False, default=Qwen/Qwen2.5-32B-Instruct)"
    echo "--embedding_model <model>    Embedding model to use for sentence encoding (required=False, default=ncbi/MedCPT-Query-Encoder)"
    echo "Options:"
    
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p|--patients_path) patients_path="$2"; shift 2 ;;
        -d|--db_path) db_path="$2"; shift 2 ;;
        -c|--clinical_trials_path) clinical_trials_path="$2"; shift 2 ;;
        -l|--LLM_model) LLM_model="$2"; shift 2 ;;
        -e|--embedding_model) embedding_model="$2"; shift 2 ;;
        *) echo "[DifFracTion] [ERROR] Unknown parameter: $1"; usage ;;
    esac
done

LLM_model_for_patients="Qwen/Qwen1.5-14B-Chat"
sentence_tokenizer="en_core_sci_sm"

LLM_model=${LLM_model:-"Qwen/Qwen2.5-32B-Instruct"}
embedding_model=${embedding_model:-"ncbi/MedCPT-Query-Encoder"}


if [[ -z "$patients_path" || -z "$db_path" ]]; then
    echo "[OncoMatch] [ERROR] Missing required parameters."
    usage
fi
# Preprocessing 

mkdir -p ${db_path}
# Run Patients
echo "[OncoMatch] [INFO] Submitting patients preprocessing job..."
job_patients=$(sbatch --parsable $(realpath ./exc/patients.sh) \
    $(realpath ${patients_path}) \
    $(realpath ${db_path}) \
    ${LLM_model_for_patients} \
    ${embedding_model} \
    ${sentence_tokenizer} \
)
# Run Trials
echo "[OncoMatch] [INFO] Submitting trials preprocessing job..."
job_trials=$(sbatch --parsable $(realpath ./exc/trials.sh) \
    $(realpath ${clinical_trials_path}) \
    $(realpath ${db_path}) \
    ${LLM_model} \
    ${embedding_model} \
    ${sentence_tokenizer} \
)

#Run interrogation
echo "[OncoMatch] [INFO] Submitting interrogation job..."
job_interrogate=$(sbatch --parsable --dependency=afterok:${job_patients}:${job_trials} $(realpath ./exc/interrogate.sh) \
    $(realpath ${db_path}/processed_trials.pkl) \
    $(realpath ${db_path}/chromaDB_patients) \
     ${LLM_model} ${embedding_model})

job_scoring=$(sbatch --parsable --dependency=afterok:${job_interrogate} $(realpath ./exc/scoring.sh) \
    $(realpath ${db_path}/chromaDB_patients) \
    $(realpath ${db_path}/processed_trials.pkl)
)

echo "[OncoMatch] Summary of submitted jobs:"
echo "[OncoMatch] - Patients job: $job_patients"
echo "[OncoMatch] - Trials job: $job_trials"
echo "[OncoMatch] - Interrogation job: $job_interrogate"
echo "[OncoMatch] - Scoring job: $job_scoring"

echo "[OncoMatch] [INFO] All jobs submitted successfully. Monitor the logs for progress."
echo "[OncoMatch] Expected output files:"
echo "[OncoMatch] From patients and trials preprocessing:"
echo "[OncoMatch] - Patients database: ${db_path}/chromaDB_patients"
echo "[OncoMatch] - Processed patients file: ${db_path}/processed_patients.pkl"
echo "[OncoMatch] - Clinical trials file: ${db_path}/processed_trials.pkl"
echo " ------------- "
echo "[OncoMatch] From interrogation:"
echo "[OncoMatch] - Interrogation datastructure: ${db_path}/PatientwTrials.pkl"
echo "[OncoMatch] - Interrogation results: ${db_path}/final_answers/FinalPatientTrialSummaries.pkl"
echo " ------------- "
echo "[OncoMatch] From scoring:"
echo "[OncoMatch] - Scoring results: ${db_path}/final_answers/scoring_results/AllPatientTrialSummariesScores.pkl"
echo " ------------- "

