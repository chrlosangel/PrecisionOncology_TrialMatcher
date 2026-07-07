# import xml.etree.ElementTree as ET  # not used
# import json                          # not used
# import re                            # not used
from pathlib import Path
# from enum import Enum                # not used
# from dataclasses import dataclass    # not used
# from typing import Union, List, Optional, Tuple  # not used
# from re import Match                 # not used
# from ast import pattern              # not used

# import numpy as np                   # not used
# import pandas as pd                  # not used

import sys, os

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


import preprocessing.utils.biomarker_CiViC_filter as biomarker_CiViC_filter   
import preprocessing.utils.parsing as parsing_ClinicalTrials
import preprocessing.utils.prompt_call as prompt_call_ClinicalTrials
import preprocessing.utils.embeddings as embeddings_ClinicalTrials

import argparse

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
     parser=argparse.ArgumentParser(description="Generate Clinical Trials Questions.")
     parser.add_argument("--clinical_trials_path", required=True, help="Path to the clinical trials path. Must include xml files.")
     parser.add_argument("--cancer_filter", default=True, help="Filter by cancer synonyms.")
     parser.add_argument("--filter_civic", default=False, help="Filter by CiViC data.")
     parser.add_argument("--LLM_model",
          default="Qwen/Qwen1.5-14B-Chat",
          choices=AVAILABLE_MODELS,
          help=f"LLM model to use for prompt generation. Choices: {', '.join(AVAILABLE_MODELS)}"
     ) 
     # ====== Embedding model arguments ======
     parser.add_argument("--embedding_model", default="ncbi/MedCPT-Query-Encoder", help="Embedding model to use for generating embeddings.")
     # ====== Save directory arguments ======
     parser.add_argument("--save_dir", required=True, help="Directory to save the ChromaDB collection and processed trials.") #---
     return parser

# To do, find a way to update this overnight, maybe with a cron job or something similar. For now, we will just download the file manually and place it in the data folder.

def main():
     parser = argument_parser()
     args = parser.parse_args()
     try:
          cuda_available = torch.cuda.is_available()
          if not cuda_available:
               print("CUDA is not available. Please ensure you have a compatible GPU and the necessary drivers installed.")
               sys.exit(1)
          print("CUDA available:", cuda_available)
          filter_words = None
     except Exception as e:
          print(f"Error checking CUDA availability: {e}")
          sys.exit(1)
     try:
          if args.filter_civic:
               path="../../../data/CiViC/nightly-VariantSummaries.tsv"
               civic, civic_df, expanded_df,synonyms = biomarker_CiViC_filter.main_(path, type_analysis="CiViC")
               print("CiViC filtering completed.")
          elif args.cancer_filter:
               print("Cancer keywords filtering applied.")
               filter_words = ["cancer", "tumor", "neoplasm"]   
          else:
               print("No filtering applied.")
          # The problem right now is that we are just providing one path, when sometimes we can have a full directory with more subs
          trials = parsing_ClinicalTrials.process_clinical_trials(args.clinical_trials_path, 
                                                                  word_filter=filter_words)

          #---- Prompts
          dnf_prompt_dir = Path("../prompts/trialQuestionsPrompt.py")
          PROMPT_DNF_TEMPLATE = prompt_call_ClinicalTrials.load_DNF_prompt(dnf_prompt_dir)

          # Initialize the LLM model
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
                    max_model_len=config['max_context']
               )
               print(f"LLM model '{model}' initialized successfully.")
               try:
                    for trial in tqdm(trials, desc="Generating DNF", unit="trial", dynamic_ncols=True):
                         prompt_call_ClinicalTrials.generate_DNF(trial, PROMPT_DNF_TEMPLATE, config, type_run='hpc', llm=llm)
              
               except Exception as e:
                    print(f"Error during DNF generation: {e}")
                    sys.exit(1)

     
          except Exception as e:
               print(f"Error initializing LLM model: {e}")
               sys.exit(1)
          
          # Save processed trials to a pickle file for later use
          embeddings_ClinicalTrials.save_questions_to_pickle(trials,Path(args.save_dir))
          
          
          
          #---- Embeddings and ChromaDB - NOT USED
          #tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)
          #tokenizer_model = AutoModel.from_pretrained(args.embedding_model)
          #client = embeddings_ClinicalTrials.generate_chromaDB_CT(trials, tokenizer, tokenizer_model, Path(args.save_dir))

     except Exception as e:
          print(f"An error occurred: {e}")
          sys.exit(1)

if __name__ == "__main__":
    main()