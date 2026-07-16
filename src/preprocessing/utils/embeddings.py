import json
import sys
import chromadb
from chromadb import Collection
import pickle
from typing import List, Optional
# from typing import Union, Tuple      # not used

from transformers import AutoTokenizer, AutoModel
# import logging                       # not used
# import openai                        # not used
import re
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
# from enum import Enum                # not used


from . import parsing as parsing_ClinicalTrials

# DataClass
ClinicalTrial=parsing_ClinicalTrials.ClinicalTrial

print("CUDA available:", torch.cuda.is_available())


def embed_text(text: str,tokenizer,model) -> np.ndarray:
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

def _parse_questions_from_json(trial:ClinicalTrial) -> List[str]:
	data = json.loads("\n".join(trial.dnf_representation))
	print(data)
	questions = []
	for i,q in data.get("QUESTIONS", []).items():
		questions.append(q)
	dnf = data.get("DNF_LOGICAL_EXPRESSION", None)
	return questions, dnf

def _add_questions(trial:ClinicalTrial,  tokenizer: AutoTokenizer, model: AutoModel,collection: Collection) -> None:
	'''Embed questions from a trial and add them to the specified ChromaDB collection.
	:param trial: ClinicalTrial - the trial object containing the questions to be embedded 
	:param tokenizer: AutoTokenizer - the tokenizer to use for embedding :param model: AutoModel - the model to use for embedding
	:param collection: Collection - the ChromaDB collection to which the embedded questions will be added
	:return: None'''

	threshold = 0.96
	embeddings,documents,metadata,ids = [],[],[],[]
	questions, dnf = _parse_questions_from_json(trial)
	for i, q in tqdm(enumerate(questions), total=len(questions), desc="Processing questions", unit="question", dynamic_ncols=True):
		e = embed_text(q, tokenizer, model)
		#print(f"Embedding for question {i}: {e}")
		if len(embeddings) > 0 : 
			existing_embeddings = np.array(embeddings)
			
			norm_new =e / np.linalg.norm(e)
			norm_existing = existing_embeddings/ np.linalg.norm(existing_embeddings, axis=1, keepdims=True)
			cosine_similarities = norm_existing @ norm_new
			if np.any(cosine_similarities > threshold):
				continue  # Skip adding this question if it's too similar to existing ones
		embeddings.append(e.tolist())
		documents.append(q)
		metadata.append({
			"trial_id": trial.name_id
			})
		ids.append(f"{trial.name_id}_Q{i}")
	if embeddings:
		print(f"Adding {len(embeddings)} questions to ChromaDB collection.")
		collection.add(ids=ids, documents=documents, metadatas=metadata, embeddings=embeddings)

def save_questions_to_pickle(processed_trials: List[ClinicalTrial], save_dir:str) -> None:


	# Make directory if it doesn't exist
	save_path = Path(save_dir)
	if not save_path.exists():
		save_path.mkdir(parents=True, exist_ok=True)
	
	# Save the processed trials to a pickle file
	if (save_path / f"processed_trials.pkl").exists():
		print(f"Adding new Clinical Trials to existing dataset.")

		sys.modules.setdefault("src.preprocessing.utils.parsing", parsing_ClinicalTrials)
		with open(save_path / f"processed_trials.pkl", "rb") as f:
			existing_trials = pickle.load(f)
		existing_trials_ids = {trial.name_id for trial in existing_trials}
		new_trials = [trial for trial in processed_trials if trial.name_id not in existing_trials_ids]
		if new_trials:
			combined_trials = existing_trials + new_trials
			with open(save_path / f"processed_trials.pkl", "wb") as f:
				pickle.dump(combined_trials, f)
			print(f"Added {len(new_trials)} new trials to the existing dataset.")
	else:
		with open(save_path / f"processed_trials.pkl", "wb") as f:
			pickle.dump(processed_trials, f) 
		print("Processed trials saved to a new pickle file.")
		
	print(f"Processed trials saved to {save_path / f'processed_trials.pkl'}")

def generate_chromaDB_CT(processed_trials: List[ClinicalTrial], 
				  tokenizer:AutoTokenizer, 
				  model: AutoModel,
				  save_dir:str) -> chromadb.Client:
	'''Generate a ChromaDB collection from a list of ClinicalTrial objects
	:param processed_trials: List[ClinicalTrial] - the list of ClinicalTrial objects to be added to the ChromaDB collection
	:param tokenizer: AutoTokenizer - the tokenizer to use for embedding, defaults to the MedCPT tokenizer
	:param model: AutoModel - the model to use for embedding, defaults to the MedCPT model
	:param save_dir: str - the directory where the ChromaDB collection will be saved
	'''
	# Create a new ChromaDB client and collection
	save_path = Path(save_dir)
	if not save_path.exists():
		save_path.mkdir(parents=True, exist_ok=True)
		
	client = chromadb.PersistentClient(path=str(save_path / "chromaDB_trials"))
	collection = client.get_or_create_collection("clinical_trials")


	existing_ids = set(collection.get()["ids"])

	# Iterate over each trial and add it to the collection
	for trial in tqdm(processed_trials, desc="Adding trials to ChromaDB", unit="trial", dynamic_ncols=True):
		if f"{trial.name_id}_Q0" in existing_ids:
			print(f"Trial {trial.name_id} already exists in the collection. Skipping.")
			continue
		try:
			_add_questions(trial, tokenizer, model, collection)
		except Exception as e:
			print(f"Error occurred while processing trial {trial.name_id}: {e}")
	return client
