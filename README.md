# Environment setup

```bash
mamba env create -f environment.yml  
conda activate trialmatch
```

# Setting up local LLMs

## Requirements

### Set up CUDA paths
vLLM requires `nvcc` for JIT compilation. Run once after creating the env:
```bash
# Find nvcc on your system
which nvcc || find /usr /opt /apps -name nvcc 2>/dev/null | head -5

# Write the activate script so paths are set automatically on conda activate
mkdir -p ~/.conda/envs/trialmatch/etc/conda/activate.d/
cat > ~/.conda/envs/trialmatch/etc/conda/activate.d/cuda_env.sh << 'EOF'
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}"
EOF

conda activate trialmatch  # re-activate to apply
```

###  Set up HF cache
```bash
mkdir -p hugging-face
HF_DIR=$(realpath ./hugging-face)
export HF_HOME=${HF_DIR}

echo "export HF_HOME=${HF_DIR}" >> ~/.bashrc
source  ~/.bashrc

#pip uninstall huggingface_hub -y --break-system-packages other installation was being executed causing issues
```

Before downloading log into huggingface and add your token make sure your token has Git permissions

```bash

hf auth login

```
Stored tokens get saved on \${HF_HOME}/stored_tokens and \${HF_HOME}/token

Follow the model instructions for downloading any HF LLM:

```bash
hf download MODEL_NAME/HF_ID --cache-dir $HF_HOME
```
# Execution
Once the environment and the proper requirements have been installed you can run the main process. It is recommended to have ALL patients and ALL trials in their respective folders (one folder per trial and one per patient). This section is to be executed by the user, where all paths can be specified, however if an automated nightly run is desired check Nighly Processing section.
## Run process 
```bash
sbatch main.sh \
     --patients_path $(realpath "./data/coral/toy_set/") \
     --db_path $(realpath "./database/chromadb/") \
     --clinical_trials_path $(realpath "./data/synthetic/trials/coral_new/") \
     --LLM_model "Qwen/Qwen2.5-32B-Instruct" \
     --embedding_model "ncbi/MedCPT-Query-Encoder"
```

The database and processing is scheduled to run every night, using cron. If not available on your system check details

## Nightly processing
This section hardcodes all files please edit scheduler.sh with your desired paths. For example, if you have a new path with new trials, you can either move all trials to the harcoded path (recommended) or change the path to the new one. 

[WARNING] cron does not accept relative paths, absolute paths are required.

If setting up for the first time on your system:

```bash

chmod +X scheduler.sh
crontab -e # opens cron, copy and paste the following line
# Cron does not accept relative paths, please modify with your absolute path [assumming you are on precision_oncology dir]
# /your/path/ = /users/jcc2340/g2lab/projects/precision_oncology <- example

0 23 * * 0,1,2,3,4 /your/path/scheduler >> /your/path/logs/cron.log >> 2&1

```

# Display patient best n results

If your data was processed correctly you can retrieve the top n trials for any given patient. If the patient is not found, please confirm the previous process was executed successfully and that the patient notes are actually in the specified folder.
Note: The patient ID is the name of the patient file record without the extension.

```bash
bash OncoMatch --patient_id "note_03_colorectal" --top_trials 3
```
