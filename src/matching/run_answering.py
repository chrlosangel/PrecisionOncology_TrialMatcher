import argparse
import sys, os
from pathlib import Path
import pickle
import logging
import torch.multiprocessing as mp
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModel,AutoConfig
mp.set_start_method('spawn', force=True)

logging.basicConfig(level=logging.INFO)
logging.getLogger("vllm").setLevel(logging.WARNING)


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matching.utils.retrieval as retrieval
import matching.utils.answering as answering

AVAILABLE_MODELS = [
    "axiong/PMC_LLaMA_13B",
    "epfl-llm/meditron-7b",
    "johnsnowlabs/JSL-MedLlama-3-8B-v2.0",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "NousResearch/Hermes-2-Pro-Mistral-7B",
    "Qwen/Qwen-14B-Chat",
    "Qwen/Qwen1.5-14B-Chat",
    "Qwen/Qwen2.5-7B-Instruct",
]

def argument_parser():
     parser=argparse.ArgumentParser(description="Retrieve patient and trial information for answering.")
     parser.add_argument("--patients_database_path", required=True, help="Path to the patients database. This is a ChromaDB database. If generated with run_patients.py it should be named chromaDB_patients.")
     parser.add_argument("--clinical_trials_file", required=True, help="Path to the clinical trials file. This is a pickle file containing the processed clinical trials.")
     parser.add_argument("--embedding_model", default="ncbi/MedCPT-Query-Encoder", help="Embedding model to use for generating question embeddings. Must be the same used in run_patients.py")
     
     #parser.add_argument("--LLM_model",
     #     default="Qwen/Qwen1.5-14B-Chat",
     #     choices=AVAILABLE_MODELS,
     #     help=f"LLM model to use for prompt generation. Choices: {', '.join(AVAILABLE_MODELS)}"
     #)
     return parser

def main():
     parser = argument_parser()
     args = parser.parse_args()

     patients_database_path = Path(args.patients_database_path).resolve()
     clinical_trials_file = Path(args.clinical_trials_file).resolve()
     save_dir = clinical_trials_file.parent

     if clinical_trials_file.suffix != ".pkl":
          raise ValueError(f"Clinical trials file '{clinical_trials_file}' is not a pickle file.")
          sys.exit(1)
     if not clinical_trials_file.exists():
          raise FileNotFoundError(f"Clinical trials file '{clinical_trials_file}' does not exist. Please provide a valid path to the processed clinical trials pickle file.")
          sys.exit(1)
     if not patients_database_path.exists():
          raise FileNotFoundError(f"Patients database path '{patients_database_path}' does not exist. Please provide a valid path to the ChromaDB database.")
          sys.exit(1)

     patients_file = (patients_database_path.parent / "processed_patients.pkl").resolve()
     
     if not patients_file.exists():
          raise FileNotFoundError(f"We couldn't find the processed patients pickle file at '{patients_file}'. Please ensure that the ChromaDB database was generated with run_patients.py and that the processed patients pickle file is present in the parent directory as the ChromaDB database.")
          sys.exit(1)
     
     patients_DB, processed_patients, processed_trials = retrieval._process_files(patients_database_path, 
                                                                                  clinical_trials_file)
     
     tokenizer_emb = AutoTokenizer.from_pretrained(args.embedding_model)
     tokenizer_model = AutoModel.from_pretrained(args.embedding_model)

     try:
          results=retrieval.process_patients_with_trials(patients_DB, 
                                            processed_patients, 
                                            processed_trials, 
                                            tokenizer_emb, 
                                            tokenizer_model,
                                            save_dir)
          
          retrieval._save_results_to_pickle(results, save_dir)

     except Exception as e:
          print(f"Error during processing patients with trials: {e}")
          sys.exit(1)
     

     #Answering section
     answer = False
     if answer:
          try:
               # This is exactly what results has, but we load it again to ensure that the file is present and can be loaded without issues
               FinalPatientsResults = (save_dir / "PatientwTrials.pkl").resolve()
               if not FinalPatientsResults.exists():
                    raise FileNotFoundError(f"Final patients results file '{FinalPatientsResults}' does not exist. Please ensure that the processing step completed successfully.")
                    sys.exit(1)
               with open(FinalPatientsResults, "rb") as f:
                    FinalPatientsResults = pickle.load(f)

               # Load template 
               answering_template_path = Path(__file__).resolve().parent.parent / "prompts" / "answeringPrompt.py"
               ANSWERING_PROMPT_TEMPLATE = answering.load_prompt(answering_template_path)     

               # Initialize LLM and configuration
               model = args.LLM_model
               if model == "epfl-llm/meditron-7b":
                    print("[WARNING]'epfl-llm/meditron-7b' context window is limited to 2048 tokens. Consider using a model with a larger context window for better performance.")
                    print("[WARNING]Input text may be truncated if it exceeds the context window size.")


               config= {
               	'model_name': model,
               	'temperature': 0.0,
                    'max_tokens': 1024,
                    'max_context': 0,
               }

               def_conf=AutoConfig.from_pretrained(config['model_name'])
               config['max_context'] = getattr(def_conf, "max_position_embeddings", None)

               try:
                    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
                    llm = LLM(
                        model=config['model_name'],
                        gpu_memory_utilization=0.88,
                        dtype='bfloat16',
                        max_model_len=13472
                    )
               except Exception as e:
                    print(f"Error initializing LLM: {e}")
                    sys.exit(1)
               
               # Generate responses
          except Exception as e:
               print(f"Error generating responses: {e}")
               sys.exit(1)


     print(f"Process completed successfully.")
if __name__ == "__main__":
     main()
     