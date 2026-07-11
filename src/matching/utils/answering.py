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
	# the json file is being wrapped around ``` so we remove those lines first
	lines = [l for l in json_response if not l.strip().startswith("```")]
	try:
		data = json.loads("\n".join(lines))
	except json.JSONDecodeError as e:
		raise ValueError(f"LLM returned invalid JSON: {e}")

	retrieved_chunks = retrieved_chunks or {}
	evidence = [
		{
			"chunk_id": e.get("CHUNK_ID"),
			"section": e.get("SECTION"),
			"chunk_text": retrieved_chunks.get(e.get("CHUNK_ID"), {}).get("CHUNK", "")
		}
		for e in data.get("EVIDENCE", [])
	]

	return PatientTrialQuestionAnswer(
		question=data.get("QUESTION", ""),
		answer=data.get("ANSWER", ""),
		confidence=int(data.get("CONFIDENCE_SCORE", 0)),
		evidence=evidence,
		answer_reasoning=data.get("ANSWER_REASONING", ""),
		question_interpretation=data.get("QUESTION_REASONING", "")
		)

def answer_patient_trials(FinalPatientsResults:List[retrieval.PatientsResults], 
				   template:str, llm:LLM, config:dict,save_dir:Path) -> List[PatientAllTrialSummaries]:
	
	max_tokens = config.get('max_tokens', 1024)
	max_context = config.get('max_context', 2048)
		
	# Reserve space for output and template overhead
	criteria_budget = (max_context - max_tokens - 200) // 2

	sampling_params = SamplingParams(temperature=config['temperature'], 
						max_tokens=max_tokens)

	os.makedirs(f"{save_dir}/final_answers/", exist_ok=True)
	
	final_answers_path = (f"{save_dir}/final_answers/" / "FinalPatientTrialSummaries.pkl").resolve()

	if final_answers_path.exists():
		with open(final_answers_path, "rb") as f:
			all_patient_summaries = pickle.load(f)
			processed = {(p.patient_id, t.trial_id) 
				for p in all_patient_summaries 
				for t in p.trial_summaries}
	else:
		all_patient_summaries = []  # Store all patient summaries
		processed = set()  # tracks (patient_id, trial_id) pairs already completed


	for p in tqdm(FinalPatientsResults.patients_results, desc="Patients"):
		if not p.trial_results: # If they are empty
			continue
		
		alltrials_patient = PatientAllTrialSummaries(patient_id=p.patient_id, trial_summaries=[])
		for t in tqdm(p.trial_results, desc=f"  Trials [{p.patient_id}]", leave=False):
			if (p.patient_id, t.trial_id) in processed:
				tqdm.write(f"  Skipping already processed: patient={p.patient_id}, trial={t.trial_id}")
				continue
			trial_prompts = []
			trial_chunks = []  # parallel list of chunks_trial per question
			for q in t.question_Results:
				chunks_trial = {}
				for i, chunk in enumerate(q.chunks):
					chunk_id = (f"chunk_{i+1}_{p.patient_id}_{t.trial_id}\n")
					chunks_trial[chunk_id] = {
						"CHUNK": f"{chunk}\n",
						"SECTION": f"{q.metadatas[i]}\n"
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
			
			# For one patient and one trial, ask LLM
			responses = llm.generate(trial_prompts, sampling_params=sampling_params)

			all_json_responses_for_trial = []  # one entry per question
			for i, response in enumerate(responses):
				# All answers for just one trial/patient combination 
				json_responses = response.outputs[0].text.strip().split("\n")
				all_json_responses_for_trial.append(json_responses)
			
			final_p_t=PatientTrialSummary(trial_id=t.trial_id, question_answers=[])
			
			for i, json_response in enumerate(all_json_responses_for_trial):
				question_answer = _parse_answer(json_response, retrieved_chunks=trial_chunks[i])
				final_p_t.question_answers.append(question_answer)

			processed.add((p.patient_id, t.trial_id))
			tqdm.write(f"  Done: patient={p.patient_id}, trial={t.trial_id}")
			alltrials_patient.trial_summaries.append(final_p_t)
		all_patient_summaries.append(alltrials_patient)
	return all_patient_summaries