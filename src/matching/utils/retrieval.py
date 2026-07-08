import chromadb
from chromadb import Collection
import pickle
from pathlib import Path
from typing import List, Optional
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch


def _load_patientDB(save_dir: str) -> chromadb.Client:
     """Loads the ChromaDB database for the processed patients.
     :param save_dir: Path to the directory where the ChromaDB database is saved
     :return: ChromaDB client object
     """
     save_path = Path(save_dir)

     if not save_path.exists():
         raise FileNotFoundError(f"ChromaDB directory '{save_dir}' does not exist.")
     
     client = chromadb.PersistentClient(path=str(save_path / "chromaDB_patients"))

     return client

def embed_query(text: str, tokenizer: AutoTokenizer, model: AutoModel) -> np.ndarray:
    '''Embed a string of text [chunk]
    :param text: str - the text to be embedded
	:param tokenizer: the tokenizer to use for embedding, defaults to the MedCPT tokenizer
	:param model: the model to use for embedding, defaults to the MedCPT model
    :return: numpy array representing the embedding of the input text float32
    '''
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", 
                           truncation=True, padding=True,
                           max_length=512)
        embeddings = model(**inputs).last_hidden_state[:,0,:].squeeze().numpy() # we take the embedding of the [CLS] token, which is the first token in the input sequence
    return embeddings.astype('float32')