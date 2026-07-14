import sys
import os
import pickle
from pathlib import Path
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple
import re
import json 
import logging
import torch
from tqdm import tqdm
print("CUDA available:", torch.cuda.is_available())

from vllm import LLM, SamplingParams
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

logging.basicConfig(level=logging.INFO)
logging.getLogger("vllm").setLevel(logging.WARNING)
# import retrieval module from the same package
from . import retrieval

# Dataclasses to store the results of the answers
@dataclass
class PatientTrialQuestionAnswer:
	question: str
	answer: str
	confidence: int
	evidence: list[dict]
	answer_reasoning: str
	question_interpretation: str
	
@dataclass
class PatientTrialSummary:
	trial_id: str
	trial_DNF: str
	question_answers: list[PatientTrialQuestionAnswer]

@dataclass
class PatientAllTrialSummaries:
	patient_id: str
	trial_summaries: list[PatientTrialSummary] = None


_FALLBACK_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}{{ message['content'] }}\n\n{% endif %}"
    "{% if message['role'] == 'user' %}{{ message['content'] }}\n\n{% endif %}"
    "{% endfor %}"
    "JSON output:"
)


def load_prompt(template_path: str) -> str:
	"""Loads a prompt template from a file.
	:param template_path: Path to the prompt template file
	:return: The content of the prompt template file as a string
	"""
	try:
		with open(template_path, 'r') as f:
			raw = f.read()
			raw = raw.replace('{',"{{").replace('}', '}}') #we might have some curlys that are not meant to be placeholders

			prompt_template=raw.split('"""')[1]
			prompt_template=prompt_template.replace("{{TRIAL_ID}}", "{trial_id}").replace("{{QUESTION}}", "{question}").replace("{{PATIENT_INFO}}", "{patient_info}").replace("{{RELEVANT_CHUNKS}}", "{relevant_chunks}")
			return prompt_template

	except FileNotFoundError:
		print(f"Error: The file {template_path} was not found.")
		sys.exit(1)

	return prompt_template

def _format_to_chat(prompt:str,llm:LLM):
	'''Format a prompt string into a chat message format for OpenAI API.'''
	system_content = "You are an expert in clinical trial eligibility criteria. Output only valid JSON, no extra text or explanations."
	messages = [{"role": "system", "content": system_content},
	            {"role": "user", "content": prompt}]

	model_name = llm.llm_engine.model_config.model
	hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
	try:
		chat_prompt = hf_tokenizer.apply_chat_template(messages, 
									    tokenize=False, 
									    add_generation_prompt=True)
	except Exception:
		# apply_chat_template Converts a list of dictionaries with "role" and "content" keys to a list of token ids. 
		chat_prompt = hf_tokenizer.apply_chat_template(
			messages, tokenize=False, add_generation_prompt=True,
			chat_template=_FALLBACK_TEMPLATE
		)
	return chat_prompt

def _parse_answer(json_response:List[str], retrieved_chunks:dict=None) -> PatientTrialQuestionAnswer:
	# Strip code fences and language tags (```json, ```, etc.)
	lines = [l for l in json_response if not l.strip().startswith("```")]
	text = "\n".join(lines)
	text = re.sub(r',\s*([\]}])', r'\1', text)  # strip trailing commas (LLMs often produce these)
	try:
		data = json.loads(text)
	except json.JSONDecodeError:
		# Fallback: extract the first {...} block in case of surrounding prose
		match = re.search(r'\{.*\}', text, re.DOTALL)
		if not match:
			raise ValueError(f"No JSON object found in LLM output:\n{text}")
		try:
			data = json.loads(match.group())
		except json.JSONDecodeError as e:
			raise ValueError(f"LLM returned invalid JSON: {e}\nRaw output:\n{text}")

	retrieved_chunks = retrieved_chunks or {}
	#.get(key)
	# Because we are accessing to a dictionary in retrieved_chunks and 
	# the keys are the chunk_ids, we can use .get to get whats inside
	# which is another dictionary with keys "CHUNK" and "SECTION"
	# and finally use .get again to get the value of "CHUNK" or "SECTION" or an empty string if not found
	evidence = []
	for e in data.get("EVIDENCE", []):
		if isinstance(e, str):
			chunk_id = e
			section = ""
		else:
			chunk_id = e.get("CHUNK_ID")
			section = e.get("SECTION", "")
		evidence.append({
			"chunk_id": chunk_id,
			"section": section,
			"chunk_text": retrieved_chunks.get(chunk_id, {}).get("CHUNK", "")
		})

	return PatientTrialQuestionAnswer(
		question=data.get("QUESTION", ""),
		answer=data.get("ANSWER", ""),
		confidence=int(data.get("CONFIDENCE_SCORE", 0)),
		evidence=evidence,
		answer_reasoning=data.get("ANSWER_REASONING", ""),
		question_interpretation=data.get("QUESTION_REASONING", "")
		)


def save_patient_summaries(all_patient_summaries:List[PatientAllTrialSummaries], save_dir:Path):
	"""Saves the patient summaries to a pickle file.
	:param all_patient_summaries: List of PatientAllTrialSummaries objects
	:param save_dir: Directory where the pickle file will be saved
	"""
	os.makedirs(save_dir, exist_ok=True)
	final_answers_path = (save_dir / "FinalPatientTrialSummaries.pkl").resolve()
	with open(final_answers_path, "wb") as f:
		pickle.dump(all_patient_summaries, f)
	print(f"Patient summaries saved to {final_answers_path}")

def answer_patient_trials(FinalPatientsResults:List[retrieval.PatientsResults], 
				   template:str, llm:LLM, config:dict,save_dir:Path) -> List[PatientAllTrialSummaries]:
	
	max_tokens = config.get('max_tokens', 1024)
	max_context = config.get('max_context', 2048)
		
	# Reserve space for output and template overhead
	criteria_budget = (max_context - max_tokens - 200) // 2

	sampling_params = SamplingParams(temperature=config['temperature'], 
						max_tokens=max_tokens)

	os.makedirs(f"{save_dir}/final_answers/", exist_ok=True)
	
	new_dir = Path(f"{save_dir}/final_answers/")
	final_answers_path = (new_dir / "FinalPatientTrialSummaries.pkl").resolve()
	
	if final_answers_path.exists():
		with open(final_answers_path, "rb") as f:
			all_patient_summaries = pickle.load(f)
			processed = {(p.patient_id, t.trial_id) 
				for p in all_patient_summaries 
				for t in p.trial_summaries}
	else:
		all_patient_summaries = []  # store all patient summaries
		processed = set()  # tracks (patient_id, trial_id) pairs already completed


	for p in tqdm(FinalPatientsResults.patients_results, desc="Patients"):
		if not p.trial_results: # If they are empty
			continue
		
		#None = the patient is new
		
		existing_patient = next((s for s in all_patient_summaries if s.patient_id == p.patient_id), None)

		# if it already exists in our summaries
		if existing_patient is not None:
			alltrials_patient = existing_patient #is a PatientAllTrialSummaries object
		else:
			alltrials_patient = PatientAllTrialSummaries(patient_id=p.patient_id, trial_summaries=[])
		new_trials_added = False
		for t in tqdm(p.trial_results, desc=f"  Trials [{p.patient_id}]", leave=False):
			trial_dnf = t.DNF if hasattr(t, 'DNF') else None
			if (p.patient_id, t.trial_id) in processed:
				tqdm.write(f"  Skipping already processed: patient={p.patient_id}, trial={t.trial_id}")
				continue
			trial_prompts = []
			trial_chunks = []  # parallel list of chunks_trial per question
			for q in t.question_Results:
				chunks_trial = {}
				for i, chunk in enumerate(q.chunks):
					chunk_id = (f"chunk_{i+1}_{p.patient_id}_{t.trial_id}")
					#removed \n because when we tried to access with the keyys, '\n' was interfering with the access
					chunks_trial[chunk_id] = {
						"CHUNK": f"{chunk}",
						"SECTION": f"{q.metadatas[i]}"
					}

				q_prompt = template.format(
					question=q.question,
					trial_id=t.trial_id,
					patient_info=p.patient_info,
					relevant_chunks=chunks_trial
				)
				q_prompt = _format_to_chat(q_prompt, llm)
				trial_prompts.append(q_prompt) #append the current question prompt
				trial_chunks.append(chunks_trial) #append the current question's chunks
			
			# Trial_prompts is a list of prompts for each question in the trial
			# Parellely, trial_chunks is a list of dictionaries (which we can assume are on the same order as the propms per question) where each dictionary has the chunk_ids as keys and the values are a dictionary with keys "CHUNK" and "SECTION"
			responses = llm.generate(trial_prompts, sampling_params=sampling_params)

			all_json_responses_for_trial = []  # one entry per question
			# now, responses should be a list same length as trial_prompts, where each entry is the LLM's response for that question
			# all responses are stored in all_json_responses_for_trial
			for i, response in enumerate(responses):
				# All answers for just one trial/patient combination 
				json_responses = response.outputs[0].text.strip().split("\n")
				all_json_responses_for_trial.append(json_responses)
			
			final_p_t=PatientTrialSummary(trial_id=t.trial_id, trial_DNF=trial_dnf, question_answers=[])
			
			# Because responses,prompts and trial_chunks are all parallel lists,we can use the index i
			# to access the corresponding questions answers/jsons and the corresponding trial_chunks for that question
			for i, json_response in enumerate(all_json_responses_for_trial):
				# json_response is expected to be a json string, where each line contains a key and a value, the keys are those we include in the
				print("Raw JSON:", json_response)
				question_answer = _parse_answer(json_response, retrieved_chunks=trial_chunks[i])
				final_p_t.question_answers.append(question_answer)

			processed.add((p.patient_id, t.trial_id))
			tqdm.write(f"  Done: patient={p.patient_id}, trial={t.trial_id}")

			#The new trial is automatically added  to th existing object
			alltrials_patient.trial_summaries.append(final_p_t) # if the patient was already existing here we append to the existing trial_summaries, because
			# alltrials_patient is already the existing patient object, by appending we are just adding a new entry
			new_trials_added = True

		if new_trials_added:
			if existing_patient is None: #if its a new patient
				all_patient_summaries.append(alltrials_patient) #create the new list of PatientAllTrialSummaries just if the patient is new
			with open(final_answers_path, "wb") as f: # here we will overwrite the patient if its already existing with the new entries 
				pickle.dump(all_patient_summaries, f)
			tqdm.write(f"  Checkpoint saved for patient={p.patient_id}")
	return all_patient_summaries