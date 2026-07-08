import sys
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

import chromadb
from chromadb import Collection
import pickle
from typing import List

from . import parsingPatients as parsingPatients
Patient=parsingPatients.Patient

# Information retrieval Contrastively Pretrained Transformer model for zero-shot semantic IR in biomedicine
# Bi encoder architecture, first retriever (query encoder and document encoder), second re-ranker.
# It was trained on 255M query-article pairs from Pubmed search logs. Real query + the article the user clicked on.
	#the two encoders are trained togheter so their outputs land in a shared embeding space (the same query is also fed with do´cuments that the user did not click on, so the model learns to distinguish between relevant and non-relevant documents)
     #Because this happens at a massive scale the model learns to scale achross real miomedical behavior, the resulting embedings tend to organize around genuine biomedical relationshipds
    # The text is first tokenized and then passed to the model 
# The second component or reranker takes the top candidates retrieved by the encoder (embedding similaruty) and re-scores them more precisely using token level interactions.  It is trained with
# the negative distrivution  sampled from the pretrained MedCPT retriever
# The retriever takes out all "Keyword-match article" because is relatively easy to match them, it needs to understand the semantic understanding of the query and the document to be able to match them. The reranker is trained with a more difficult negative distribution, which is sampled from the pretrained MedCPT retriever. 



#the weights of those transformer layers were shaped by the contrastive training to do something specific: encode the input such that semantically/clinically related texts land near each other and unrelated ones land far apart. The "knowledge" isn't stored as a lookup of phrases to locations — it's stored in the weights of the network, which then compute an appropriate vector for any new text you give it, including text it never saw during training.
# That's the part that makes it generalize: if you feed it a clinical phrase it never saw verbatim during training, it still produces a sensible vector, because the network learned general patterns of biomedical language

#could include bert or medcpt but bert shows worse performance based on MedCPT paper 
# MedCPT: https://huggingface.co/ncbi/MedCPT-Query-Encoder
# Check alternatives on references https://academic.oup.com/bioinformatics/article/39/11/btad651/7335842

print("CUDA available:", torch.cuda.is_available())



def embed_text(text: str,tokenizer: AutoTokenizer, model: AutoModel) -> np.ndarray:
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

def _add_keywords(patient:Patient, tokenizer: AutoTokenizer, model: AutoModel,collection: Collection) -> None:
	'''embedding keywords and adding them to the ChromaDB collection for a patient
	:param patient: Patient object with extracted sections and keywords
	:param tokenizer: the tokenizer to use for embedding
	:param model: the model to use for embedding
	:param collection: ChromaDB collection to add the embeddings to'''

	threshold = 0.96
	#embedding tracks embeddings, documents tracks the text of the keyword, metadata tracks the patient id and section name, ids tracks the unique id for each keyword
	embeddings, documents,metadata,ids_list = [], [], [], []
	global_idx = 0
	try:
		for s in patient.PatientSections:
			if not s.keywords:
				continue
			for kw in tqdm(s.keywords, desc=f"Processing keywords for patient {patient.patient_id}", unit="keyword"):
				e = embed_text(kw, tokenizer, model)
				if len(embeddings) > 0:
					existin_emb = np.array(embeddings)
					norm_embedding = e / np.linalg.norm(e)
					norm_existing_embeddings = existin_emb / np.linalg.norm(existin_emb, axis=1, keepdims=True)
					cosine_similarities = norm_existing_embeddings @ norm_embedding
					if np.any(cosine_similarities > threshold):
						print(f"Skipping keyword due to similarity: {kw}")
						continue
				embeddings.append(e.tolist())
				documents.append(kw)
				metadata.append({
					"patient_id": str(patient.patient_id),
					"section": s.section_name,})
				ids_list.append(f"{patient.patient_id}_kw_{global_idx}")
				global_idx += 1
		if embeddings:
			collection.add(ids=ids_list, embeddings=embeddings, documents=documents, metadatas=metadata)

	except Exception as e:
		print(f"Error adding keywords for patient {patient.patient_id}: {e}")

				
			
def _add_chunks(patient:Patient, tokenizer: AutoTokenizer, model: AutoModel,collection: Collection) -> None:
	'''embedding chunks and adding them to the ChromaDB collection for a patient
	:param patient: Patient object with extracted sections and chunks
	:param tokenizer: the tokenizer to use for embedding
	:param model: the model to use for embedding
	:param collection: ChromaDB collection to add the embeddings to'''
	threshold = 0.96
	#embedding tracks embeddings, documents tracks the text of the chunk, metadata tracks the patient id and section name, ids tracks the unique id for each chunk
	embeddings,documents,metadata,ids_list = [], [], [], []
	global_idx = 0
	try:
		for s in patient.PatientSections:#each section for the patient
			# global_idx will stay loaded when we move to the next section. So is the total amount of chunks per patient, not per section. 
			if not s.chunks: # Each section has a list of chunks
				continue
			for chunk in tqdm(s.chunks, desc=f"Processing chunks for patient {patient.patient_id}", unit="chunk"):
				e = embed_text(chunk, tokenizer, model)
				if len(embeddings) > 0:
					existin_emb = np.array(embeddings)
					norm_embedding = e / np.linalg.norm(e)
					norm_existing_embeddings = existin_emb / np.linalg.norm(existin_emb, axis=1, keepdims=True)
					cosine_similarities = norm_existing_embeddings @ norm_embedding
					if np.any(cosine_similarities > threshold):
						print(f"Skipping chunk due to similarity: {chunk}")
						continue
				embeddings.append(e.tolist())
				documents.append(chunk)
				metadata.append({
					"patient_id": str(patient.patient_id),
					"section": s.section_name,})
				ids_list.append(f"{patient.patient_id}_chunk_{global_idx}")
				global_idx += 1
		if embeddings:
			collection.add(ids=ids_list, embeddings=embeddings, documents=documents, metadatas=metadata)

	except Exception as e:
		print(f"Error adding chunks for patient {patient.patient_id}: {e}")


def generate_chromaDB(processed_patients: List[Patient], tokenizer: AutoTokenizer, 
				  model: AutoModel, save_dir: str, option: str) -> chromadb.Client:
	"""Generates a ChromaDB database for the processed patients, storing both chunks and keywords.
	:param processed_patients: List of Patient objects with extracted sections and chunks
	:param tokenizer: the tokenizer to use for embedding
	:param model: the model to use for embedding
	:param save_dir: Path to the directory where the ChromaDB database will be saved, do not include the database name, just the directory
	:param option: String indicating the option for vector store creation, it can be 'chunks'. 'keywords' or 'both'

	The chromaDB database will contain on save_dir/chromaDB_patients: 
	- chunks collection with embeddings and metadata for each chunk (chunk text + embedding + patient id + section name)
	- keywords collection with embeddings and metadata for each keyword (keyword text + embedding + patient id + section name)
	"""
	save_path = Path(save_dir)
	if not save_path.exists():
		save_path.mkdir(parents=True, exist_ok=True)
	client = chromadb.PersistentClient(path=str(save_path / "chromaDB_patients"))
	
	if option not in ["chunks", "keywords", "both"]:
		raise ValueError("Invalid option. Must be 'chunks', 'keywords', or 'both'.")
	if option in ["chunks", "both"]:
          #Setting of collection metadata is set to {"hnsw:space": "ip"} to use inner product (cosine similarity) for nearest neighbor search, which is appropriate for embeddings
		chunks_collection = client.get_or_create_collection("chunks", metadata={"hnsw:space": "ip"})
		print(f"Created ChromaDB collection for chunks at {save_path / 'chromaDB_patients' / 'chunks'}")
	if option in ["keywords", "both"]:
		keywords_collection = client.get_or_create_collection("keywords", metadata={"hnsw:space": "ip"})
		print(f"Created ChromaDB collection for keywords at {save_path / 'chromaDB_patients' / 'keywords'}")

	for patient in tqdm(processed_patients, desc="Building ChromaDB", unit="patient"):
		pid = str(patient.patient_id)
		# Check if patient already exists using metadata
		if option in ["chunks", "both"]:
			# where= filters by metadata 
			# because we used one id per patient we can get all the info with that ID
			# and ['ids'] is just added to retrieve all chunks for that patient
			# is the list is empty, it means we did not add any chunks for the patient and therefore it will go to the next step
			if chunks_collection.get(where={"patient_id": pid})["ids"]:
				print(f"Chunk {pid} already exists in chunks collection. Skipping.")
				continue
		if option in ["keywords", "both"]:
			if keywords_collection.get(where={"patient_id": pid})["ids"]:
				print(f"Keyword {pid} already exists in keywords collection. Skipping.")
				continue

		if option in ["keywords", "both"]:
			print(f"Adding keywords for patient {patient.patient_id} to ChromaDB collection.")
			_add_keywords(patient, tokenizer, model, keywords_collection)
		if option in ["chunks", "both"]:
			print(f"Adding chunks for patient {patient.patient_id} to ChromaDB collection.")
			_add_chunks(patient, tokenizer, model, chunks_collection)
	print(f"ChromaDB database generated at {save_path / 'chromaDB_patients'}")
	return client

def save_patients_to_pickle(processed_patients: List[Patient], save_dir:str) -> None:
	# Need to change for version were we only have one file and patients already included are not added

	# Make directory if it doesn't exist

	save_path = Path(save_dir)

	#Main file
	patients_pkl = save_path / f"processed_patients.pkl"

	if not save_path.exists():
		save_path.mkdir(parents=True, exist_ok=True)
	

	if patients_pkl.exists():
		print("Adding new patients to existing dataset.")
		#Added this because we saved a pickle Patient class with the first one, but now we call it with the second
		# It will work only if the saved pkl was generated with the same Patient class, otherwise it will fail.
		sys.modules.setdefault("src.preprocessing.utils.parsingPatients", parsingPatients)
		
		with open(patients_pkl, "rb") as f:
			existing_patients = pickle.load(f)

		existing_patients_ids=[p.patient_id for p in existing_patients]
		new_patients = [p for p in processed_patients if p.patient_id not in existing_patients_ids]
		if new_patients:
			print(f"Adding {len(new_patients)} new patients to existing dataset.")
			with open(patients_pkl, "wb") as f:
				pickle.dump(existing_patients + new_patients, f)
		if not new_patients:
			print("No new patients to add. Existing dataset remains unchanged.")

	else:
		with open(patients_pkl, "wb") as f:
			pickle.dump(processed_patients, f)
		print(f"Processed patients saved to {patients_pkl}")
		
	print(f"Processed patients saved to {save_path / f'processed_patients.pkl'}")



### Faiss index creation and saving/loading functions 
# Might go to graveyard if we choose to use ChromaDB

#class PatientVectorStore:
#	"""Class for storing patient embeddings and metadata in a FAISS index."""
#	index: faiss.IndexFlatIP
#	metadata: List[dict]
#	n_chunks: int
	
#def build_vector_stores_for_patient(patient:Patient, tokenizer: AutoTokenizer,model: AutoModel,option: str) -> dict:
#	'''Builds a vector store for a patient by embedding their clinical note sections and storing the embeddings in a FAISS index.
#	:param patient: Patient object with extracted sections and chunks
#	:param tokenizer: the tokenizer to use for embedding
#	:param model: the model to use for embedding
#	:param option: String indicating the option for vector store creation, it can be chunks or keywords
#	:return: index and metadata for the vector store based on the specified option'''
#	''' One index per patient, which includes multiple embeddings (one per chunk or keyword)
#	If a patient has 3 sections with 2 chunks each we should have 6 embeddings in the index, and the metadata should reflect which section and chunk each embedding corresponds to, for example:
#	index vector 0  →  chunk_metadata[0] = {patient_id, section: "past medical history", chunk: "..."}
#	index vector 1  →  chunk_metadata[1] = {patient_id, section: "past medical history", chunk: "..."}
#	index vector 2  →  chunk_metadata[2] = {patient_id, section: "family history", chunk: "..."}
#	...
#	'''
#	
#	threshold,dimension = 0.96,768
#	patient_embeddings = []
#	if option == 'chunks':
#		chunk_metadata = [] # we can keep track of the metadata for each chunk, such as the section it came from and the original text, to help with debugging and analysis later on
#		try:
#			index = faiss.IndexFlatIP(dimension)
#			for section in patient.PatientSections:
#				for chunk in section.chunks:
#					embedding = embed_text(chunk, tokenizer, model) #chunks are completely filtered before this step
#					if len(patient_embeddings) > 0:
#						existing_embeddings = np.array(patient_embeddings)
#						norm_embedding = embedding / np.linalg.norm(embedding)
#						norm_existing_embeddings = existing_embeddings / np.linalg.norm(existing_embeddings, axis=1, keepdims=True)
#						cosine_similarities = norm_existing_embeddings @ norm_embedding
#						if np.any(cosine_similarities > threshold):
#							print("Skipping chunk due to similarity")
#							continue
#					patient_embeddings.append(embedding)
#					#What information are we keeping track of?
#					chunk_metadata.append({
#						"patient_id": patient.patient_id,
#						"section": section.section_name,
#						"chunk": chunk,
#					})
#			if patient_embeddings:
#				patient_embeddings = np.array(patient_embeddings)
#				index.add(patient_embeddings)
#				
#			return index, chunk_metadata
#		except Exception as e:
#			print(f"Error building vector store for patient {patient.patient_id}: {e}")
#
#	elif option == 'keywords':
#		keyword_metadata = []
#		try:
#			index = faiss.IndexFlatIP(dimension)
#			for s in patient.PatientSections:
#				if s.keywords is not None:
#					for k in s.keywords:
#						embedding = embed_text(k, tokenizer, model)
#						if len(patient_embeddings) > 0:
#							existing_embeddings = np.array(patient_embeddings)
#							norm_embedding = embedding / np.linalg.norm(embedding)
#							norm_existing_embeddings = existing_embeddings / np.linalg.norm(existing_embeddings, axis=1, keepdims=True)
#							cosine_similarities = norm_existing_embeddings @ norm_embedding
#							if np.any(cosine_similarities > threshold):
#								print("Skipping keyword due to similarity")
#								continue
#						patient_embeddings.append(embedding)
#						keyword_metadata.append({
#							"patient_id": patient.patient_id,
#							"section": s.section_name,
#							"keyword": k,
#						})
#			if patient_embeddings:
#				patient_embeddings = np.array(patient_embeddings)
#				index.add(patient_embeddings)
#			return index, keyword_metadata
#		except Exception as e:
#			print(f"Error building vector store for patient {patient.patient_id}: {e}")
#			return None, None
#		