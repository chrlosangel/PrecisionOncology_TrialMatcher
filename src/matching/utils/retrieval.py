import chromadb
from chromadb import Collection
import pickle
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import sys


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))  # Add the src directory to the path
from preprocessing.utils import embeddings as embeddingTrials
# embeddingTrials contains _parse_questions_from_json


# ====== Dataclasses to store the results of the matching process ======
# Multiple of this inside TrialResult because we have multiple questions per trial
@dataclass
class QuestionResult:
    question: str
    chunks: List[str]
    distances: List[float]
    metadatas: List[dict]

# Multiple of this per patient because we have multiple trials per patient
@dataclass
class TrialResult:
    trial_id: str
    DNF: str
    question_Results: List[QuestionResult] = field(default_factory=list)

@dataclass
class PatientMatchingTrials:
    patient_id: str
    patient_info: Optional[dict] = None
    trial_results: List[TrialResult] = field(default_factory=list)

@dataclass
class PatientsResults: #All patients results. we have a list per patient so just access to it as p.
    patients_results: List[PatientMatchingTrials] = field(default_factory=list)

# ======== 



def _load_patientDB(database_path: Path) -> chromadb.Client:
     """Loads the ChromaDB database for the processed patients.
     :param database_path: Path to the directory where the ChromaDB database is saved
     :return: ChromaDB client object
     """
     client = chromadb.PersistentClient(path=str(database_path))

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

def _process_files(patients_db_path: Path, clinical_trials_file: Path) -> tuple[chromadb.Client, list, list]:
    """Loads the processed patients and trials from pickle files.
    :param patients_db_path: Path to the directory where the ChromaDB database is saved
    :param clinical_trials_file: Path to the pickle file containing the processed trials
    :return: Tuple containing the processed patients and trials
    """
    # Load the processed patients from the ChromaDB database
    patient_client = _load_patientDB(patients_db_path)
    patients_file = ((patients_db_path).parent / "processed_patients.pkl").resolve()

    # Load the processed trials from the pickle file
    if patients_file.exists():
        with open(patients_file, "rb") as f:
            processed_patients = pickle.load(f)
    else:
        raise FileNotFoundError(f"Processed patients file not found at {patients_file}")

    with open(clinical_trials_file, "rb") as f:
        processed_trials = pickle.load(f)

    return patient_client, processed_patients, processed_trials

def _save_results_to_pickle(results: PatientsResults, path: Path) -> None:
    """Saves the results to a pickle file.
    :param results: PatientsResults object containing the matching results
    :param path: Path to the pickle file where the results will be saved
    """
    path = path / "PatientwTrials.pkl"
    with open(path, "wb") as f:
        pickle.dump(results, f)
    
    print(f"Patients interrogation with trials saved to {path}")
        

class _RemappedUnpickler(pickle.Unpickler):
    _MODULE_REMAP = {
        "src.matching": "matching",
        "src.preprocessing": "preprocessing",
        "src.preprocessing.utils": "preprocessing.utils",
        "src.matching.utils": "matching.utils",
        "src.preprocessing.utils.parsingPatients": "preprocessing.utils.parsingPatients",
    }

    def find_class(self, module, name):
        for old, new in self._MODULE_REMAP.items():
            if module.startswith(old):
                module = new + module[len(old):]
                break
        return super().find_class(module, name)

def _load_pickle_with_remapped_modules(f):
    return _RemappedUnpickler(f).load()

def retrieve_chunks_for_trial_questions_patientxtrial(patient_client: chromadb.Client,
                                 processed_patients: list, 
                                 processed_trials: list,
                                 tokenizer: AutoTokenizer,
                                 model: AutoModel,
                                 save_dir: Optional[Path] = None,
                                 collection: str = "chunks") -> PatientsResults:

    """Processes the patients and trials, and returns the matching results.
    :param patient_client: ChromaDB client object
    :param processed_patients: List of processed patients
    :param processed_trials: List of processed trials already containing the questions
    :param tokenizer: the tokenizer to use for embedding, defaults to the MedCPT tokenizer
    :param model: the model to use for embedding, defaults to the MedCPT model
    :param collection: Name of the ChromaDB collection to query, defaults to "chunks"
    :return: PatientsResults object containing the best chunks for each question
    """

    # Initialize the PatientsResults object to store the results
    final_results_path = save_dir / "PatientwTrials.pkl"

    if final_results_path.exists():
        print(f"Loading existing interrogation results from {final_results_path}")
        with open(final_results_path, "rb") as f:
            FinalPatientsResults = _load_pickle_with_remapped_modules(f)
    else:
        print(f"Creating new interrogation results at {final_results_path}")
        FinalPatientsResults = PatientsResults(patients_results=[])

    try:
        collection = patient_client.get_collection(collection)
    except Exception as e:
        print(f"Error accessing collection '{collection}': {e}")
        sys.exit(1)
    
    try:
        # Iterate over each patient and process their trials
        for p in tqdm(processed_patients, desc="Patients", unit="patient", dynamic_ncols=True):
            pid = p.patient_id
            patient_Age = p.age
            patient_Gender = p.gender
            patient_info = {
                "age": patient_Age,
                "gender": patient_Gender
            }

            if not collection.get(where={'patient_id': pid})['ids']:
                tqdm.write(f"Warning: Patient {pid} not found in the database. Skipping this patient.")
                continue

            # Reuse existing result if patient was already in the loaded pickle
            # Next is used to find the first match and then stop searching, returning None if no match is found
            existing = next((pmr for pmr in FinalPatientsResults.patients_results if pmr.patient_id == pid), None)
            if existing is not None:
                patient_result = existing
                is_new_patient = False
            else:
                patient_result = PatientMatchingTrials(patient_id=pid, patient_info=patient_info)
                is_new_patient = True

            tqdm.write(f"Processing patient {pid} with {len(processed_trials)} trials.")

            new_trials_added = False
            for trial in tqdm(processed_trials, desc="  Trials", unit="trial", leave=False, dynamic_ncols=True):
                #Check if we have already processed this trial for this patient
                already_processed = any(t.trial_id == trial.name_id for t in patient_result.trial_results)

                if already_processed:
                    tqdm.write(f"Trial {trial.name_id} already processed for patient {pid}. Skipping.")
                    continue

                questions, dnf = embeddingTrials._parse_questions_from_json(trial)
                dnf = dnf[0] if dnf else None  # Assuming dnf is a list and we want the first element, or None if empty
                trial_result = TrialResult(trial_id=trial.name_id, DNF=dnf) # Initialize for a given trial

                for question in tqdm(questions, desc="    Questions", unit="q", dynamic_ncols=True, leave=False):
                    question_result = QuestionResult(question=question, chunks=[], distances=[], metadatas=[]) # Initialize for a given question
                    e = embed_query(question, tokenizer, model)

                    results = collection.query(
                        query_embeddings=[e],
                        n_results=8,
                        where={"patient_id": pid},
                        include=["metadatas", "distances", "documents"]
                    )
                    question_result.chunks = results['documents'][0]
                    question_result.distances = results['distances'][0]
                    question_result.metadatas = results['metadatas'][0]

                    trial_result.question_Results.append(question_result)

                new_trials_added = True
                # Either the patient is new or already processed, we append the trial result in case the trial is new for this patient
                # Because we opened the file we are appending on memory and then we return FinalPatientsReults which will already contain the new entries
                patient_result.trial_results.append(trial_result)

            if is_new_patient:
                FinalPatientsResults.patients_results.append(patient_result)
    except Exception as e:
        print(f"An error occurred during interrogation of patients with trials: {e}")
        sys.exit(1)


    # PatientsResults object is returned, and can be saved to a pickle file if needed
    return FinalPatientsResults
