import argparse
import sys, os
from pathlib import Path
import requests
import ast
import logging
import openai
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModel,AutoConfig

mp.set_start_method('spawn', force=True)

logging.basicConfig(level=logging.INFO)
logging.getLogger("vllm").setLevel(logging.WARNING)


import preprocessing.utils.parsingPatients as parsingPatients   
import preprocessing.utils.chunkPatients as chunkPatients
import src.preprocessing.utils.keywordsPatients as keywordsPatients


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
     parser=argparse.ArgumentParser(description="Generate Patient DataBase.")
     parser.add_argument("--patients_path", required=True, help="Path to the patients notes.")
     parser.add_argument("--LLM_model",
          default="Qwen/Qwen1.5-14B-Chat",
          choices=AVAILABLE_MODELS,
          help=f"LLM model to use for prompt generation. Choices: {', '.join(AVAILABLE_MODELS)}"
     ) 
     # Optional argument for sentence tokenizer
     parser.add_argument("--sentence_tokenizer", required=False, default="en_core_sci_sm", help="Tokenizer for sentence chunking.")
     return parser


def main():
     parser = argument_parser()
     args = parser.parse_args()
     try:
          if args.sentence_tokenizer:
               import spacy
               sentence_tokenizer = spacy.load(args.sentence_tokenizer)

          else:
               import spacy
               sentence_tokenizer = spacy.load(args.sentence_tokenizer)
     except Exception as e:
          print(f"Error loading tokenizer: {e}")
          sys.exit(1)

     try:
          patients_path = args.patients_path
          if not os.path.exists(patients_path):
               raise FileNotFoundError(f"Patients path '{patients_path}' does not exist.")
          
          patients_parsed = parsingPatients.parse_raw_clinical_notes(patients_path)
          # Here we have a list of Patient class objects:,
          # the most important attibute is processed_patients[m].PatientSections
          # Which is a list of PatientSection Class objects [i, many per patient], each of which has a section_name and a list of chunks
          
          processed_patients, average_chunks_per_patient = chunkPatients.process_patients(patients_parsed,sentence_tokenizer)

          #---- Patient Section Keywords Extraction ----
          keyword_template_path = Path("../prompts/extractKeyWords.py")
          KEYWORD_PROMPT_TEMPLATE = keywordsPatients.load_KEYWORD_prompt(keyword_template_path)


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
               print(f"LLM model '{model}' initialized successfully.")
               try:
                    for p in tqdm(processed_patients, desc="Extracting Keywords", unit="patient", dynamic_ncols=True):
                         keywordsPatients.extract_patient_sections_keywords(p, n_keywords=5, template=KEYWORD_PROMPT_TEMPLATE, config=config, type_run='hpc', llm=llm)
               except Exception as e:
                    print(f"Error during keyword extraction: {e}")
                    sys.exit(1)

          except Exception as e:
               print(f"Error initializing LLM: {e}")
               sys.exit(1)

     except Exception as e:
          print(f"Error parsing clinical notes: {e}")
          sys.exit(1)
     