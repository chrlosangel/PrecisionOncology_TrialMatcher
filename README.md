## Environment setup
```bash
mamba env create -f environment.yml  # will be modified as we progress in the project

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

```bash
#Models based on PRISM paper
# https://www.nature.com/articles/s41746-024-01274-7/figures/1
# Meditron and TrialLlama have a context window of 4kb, limiting the input per patient. However out technique is emproiying only inclusion and exclusion criteria. we will get back at this. Do the amount of input per patient.
```


Before downloading log into huggingface and add your token make sure your token has Git permissions

```bash

hf auth login

```
Stored tokens get saved on \${HF_HOME}/stored_tokens and \${HF_HOME}/token

# Follow the model instructions for downloading any HF LLM
```bash
hf download MODEL_NAME/HF_ID --cache-dir $HF_HOME
```

## Models Reference

### Biomarker Extraction (Trial Side)

| Model | HF ID | Size | Paper | Role |
|---|---|---|---|---|
| Hermes-2-Pro-Mistral-7B | `NousResearch/Hermes-2-Pro-Mistral-7B` | 14 GB | Alkhoury et al. 2025 | DNF biomarker extraction from trial criteria. F2=0.94 inclusion, outperforms GPT-4 (F2=0.29) |

### Matching

| Model | HF ID | Size | Paper | Role |
|---|---|---|---|---|
| Qwen1.5-14B | `Qwen/Qwen1.5-14B-Chat` | 28 GB | PRISM / Gupta et al. | Base for OncoLLM fine-tuning. 32K context. $0.17/pair vs GPT-4 $6.18/pair |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | 14 GB | Gueguen et al. 2025 | LLM re-ranking with TrialGPT prompting. NDCG@3: 0.61 → 0.64 on real patients |
| Qwen-14B-Chat | `Qwen/Qwen-14B-Chat` | 28 GB | — | Qwen1.0 baseline. Superseded by Qwen1.5-14B (shorter 8K context) |

### Domain-Specific Baselines for matching

| Model | HF ID | Size | Paper | Role |
|---|---|---|---|---|
| Meditron-7B | `epfl-llm/meditron-7b` | 14 GB | Chen et al. 2025 | EPFL medical LLM. Pre-trained on PubMed + clinical guidelines. Requires HF agreement |
| JSL-MedLlama-3-8B | `johnsnowlabs/JSL-MedLlama-3-8B-v2.0` | 16 GB | Chen et al. 2025 | John Snow Labs medical LLM based on LLaMA-3. CC-BY-NC-ND license |
| PMC-LLaMA-13B | `axiong/PMC_LLaMA_13B` | 26 GB | — | Fine-tuned on PubMed Central. Biomedical baseline. Not the same as Trial-LLAMA |

### General Baselines

| Model | HF ID | Size | Paper | Role |
|---|---|---|---|---|
| Mistral-7B-Instruct-v0.2 | `mistralai/Mistral-7B-Instruct-v0.2` | 14 GB | Chen et al., Morrison et al. | Apache 2.0. Base model for Hermes. Strong general baseline |
| Mixtral-8x7B-Instruct | `mistralai/Mixtral-8x7B-Instruct-v0.1` | 90 GB | Chen et al., Morrison et al. | MoE (8×7B experts). Apache 2.0. Strong open-source GPT-4 alternative |
| LLaMA-3-8B-Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` | 16 GB | Morrison et al. | Meta general baseline |

### Retrieval / Embeddings

| Model | HF ID | Size | Paper | Role |
|---|---|---|---|---|
| MedCPT | `ncbi/MedCPT-Query-Encoder` | 440 MB | Jin et al. (TrialGPT) 2024 | Biomedical semantic embeddings. Trained on PubMed search logs. Used for FAISS/ChromaDB patient and trial embeddings |



# Run patients processing and database generation

```bash
conda activate trialmatch

patients=$(realpath "./data/coral/toy_set/")
db=$(realpath "./database/chromadb/")

python src/preprocessing/run_patients.py \
--patients_path ${patients} \
--LLM_model "Qwen/Qwen1.5-14B-Chat" \
--embedding_model "ncbi/MedCPT-Query-Encoder" \
--save_dir ${db} \
--sentence_tokenizer "en_core_sci_sm"

# Output ${db}/chromaDB_patients 
# - Either updated or newly generated

```
## System version Will implement parameters too

```bash
mkdir logs
sbatch test_patients.sh 
```

# Run Trials processing
```bash
ct=$(realpath "./data/synthetic/trials/coral/")
savedir=$(realpath "./database/chromadb/")

python src/preprocessing/run_trials.py \
--clinical_trials_path ${ct} \
--cancer_filter "False" \
--LLM_model "Qwen/Qwen1.5-14B-Chat" \
--embedding_model "ncbi/MedCPT-Query-Encoder" \
--save_dir ${savedir}
```

## System version Will implement parameters too
```bash
mkdir logs
sbatch test_trials_run.sh 
```
