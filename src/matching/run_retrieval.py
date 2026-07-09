import argparse
import sys, os
from pathlib import Path
import pickle
from transformers import AutoTokenizer, AutoModel,AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matching.utils.retrieval as retrieval

def argument_parser():
     parser=argparse.ArgumentParser(description="Generate Patient DataBase.")
     parser.add_argument("--patients_database_path", required=True, help="Path to the patients database. This is a ChromaDB database. If generated with run_patients.py it should be named chromaDB_patients.")
     parser.add_argument("--clinical_trials_file", required=True, help="Path to the clinical trials file. This is a pickle file containing the processed clinical trials.")
     parser.add_argument("--embedding_model", default="ncbi/MedCPT-Query-Encoder", help="Embedding model to use for generating question embeddings. Must be the same used in run_patients.py")
     return parser

def main():
     parser = argument_parser()
     args = parser.parse_args()

     patients_database_path = Path(args.patients_database_path).resolve()
     clinical_trials_file = Path(args.clinical_trials_file).resolve()
     save_dir = clinical_trials_file.parent
     print(save_dir)
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
     
     patients_DB, processed_patients, processed_trials = retrieval._process_files(patients_database_path, clinical_trials_file)
     
     tokenizer_emb = AutoTokenizer.from_pretrained(args.embedding_model)
     tokenizer_model = AutoModel.from_pretrained(args.embedding_model)

     results=retrieval.process_patients_with_trials(patients_DB, 
                                            processed_patients, processed_trials, 
                                            tokenizer_emb, tokenizer_model,save_dir)
     
     retrieval._save_results_to_pickle(results, save_dir)

     print(f"Process completed successfully.")
if __name__ == "__main__":
     main()
     