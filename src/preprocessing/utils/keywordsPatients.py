import logging
import openai
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Optional, List, Union, Tuple
import re
from pathlib import Path
import sys,os
import json 
import ast

from . import parsingPatients 

Patient=parsingPatients.Patient

print("CUDA available:", torch.cuda.is_available())

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

logging.basicConfig(level=logging.INFO)
logging.getLogger("vllm").setLevel(logging.WARNING)



_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
	     "KEYWORDS": {
		  "type": "array",
		  "items": {"type": "string"}
	   },
	    	"KEYWORD_EXTRACTION_REASONING": {"type": "string"}
    },
    "required": ["KEYWORDS", "KEYWORD_EXTRACTION_REASONING"]
}

def load_KEYWORD_prompt(template_path: str) -> str:
	"""Loads a prompt template from a file.
	:param template_path: Path to the prompt template file
	:return: The content of the prompt template file as a string
	"""
	try:
		with open(template_path, 'r') as f:
			raw = f.read()
			raw = raw.replace('{',"{{").replace('}', '}}') #we might have some curlys that are not meant to be placeholders

			prompt_template=raw.split('"""')[1]
			prompt_template=prompt_template.replace("{{input_text}}", "{input_text}").replace("{{num_keywords}}", "{num_keywords}")
			return prompt_template

	except FileNotFoundError:
		print(f"Error: The file {template_path} was not found.")
		sys.exit(1)

	return prompt_template


_FALLBACK_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}{{ message['content'] }}\n\n{% endif %}"
    "{% if message['role'] == 'user' %}{{ message['content'] }}\n\n{% endif %}"
    "{% endfor %}"
    "JSON output:"
)

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

def _parse_keywords_from_json_lines(keywords_json: List[str]) -> Optional[List[str]]:
	"""Shared parsing logic: scans vLLM output lines for the KEYWORDS field."""
	for kw in keywords_json: #The json output from the LLM is a list of srtings
		if kw and re.match(r'^\s+"KEYWORDS"', kw):
			string_kw = kw.strip().split("KEYWORDS:")[0].strip()
			splt_str = string_kw.split(":")
			k_list = splt_str[1]
			k_list = k_list[:-1]
			k_list = ast.literal_eval(k_list)
			return k_list
	return None

def _extract_keywords_from_outputs(section_jobs, outputs, p: Patient):
	for (s_n, s), output in zip(section_jobs, outputs):
		try:
			keywords_json = output.outputs[0].text.strip().split("\n") # We asked the LLM to return a JSON like structure with keywords so the whole structure is saved in keywords_json
			p.PatientSections[s_n].keywords_json = keywords_json
			p.PatientSections[s_n].keywords = _parse_keywords_from_json_lines(keywords_json) #inside the structure of the JSON we extract the keywords and save them in the PatientSection object
			print(f"Extracted keywords for section {s}: {p.PatientSections[s_n].keywords}")
		except Exception as e:
			print(f"Error extracting keywords for section {s}: {e}")
			p.PatientSections[s_n].keywords_json = []
			p.PatientSections[s_n].keywords = []
	
def _truncate_criteria(text: Optional[str], tokenizer, max_tokens: int) -> str:
	'''Truncate text to fit within max_tokens using the model tokenizer.'''
	if text is None:
		return ""
	
	#Always chop it to max_tokens/criteria_budget
	tokens = tokenizer.encode(text)
	if len(tokens) > max_tokens:
		tokens = tokens[:max_tokens]
		text = tokenizer.decode(tokens)
	return text

def extract_patient_sections_keywords(p: Patient, n_keywords: int, template: str, config: dict,
									   type_run: str = "debug", llm=None) -> None:
	"""Extracts keywords for each section of a patient's clinical note.
	:param p: Patient object containing the clinical note and processed sections
	:param n_keywords: Number of keywords to extract per section
	:param template: Prompt template string for keyword extraction loaded with load_KEYWORD_prompt
	:param config: Dictionary containing configuration parameters such as model_name and temperature
	:param type_run: 'debug' uses a local vLLM OpenAI-compatible server (localhost:8000);
	                 'hpc'   uses a pre-initialized vLLM LLM object for direct batched inference.
	:param llm: required when type_run='hpc' — a pre-initialized vllm.LLM object.
	"""
	sections = parsingPatients.split_ct_into_sections(p.description)

	max_tokens = config.get('max_tokens', 1024)
	max_context = config.get('max_context', 2048)
	
	# Reserve space for output and template overhead
	criteria_budget = (max_context - max_tokens - 200) // 2

	tokenizer = llm.get_tokenizer()

	if type_run not in ["debug", "hpc"]:
		raise ValueError(f"Unknown type_run='{type_run}'. Expected 'debug' or 'hpc'.")

	if type_run == "debug":
		client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
		s_n = 0
		for s in sections:
			section_text = sections[s]
			if section_text:
				try:
					prompt = template.format(input_text=section_text, num_keywords=n_keywords)
					response = client.chat.completions.create(
						model=config['model_name'],
						messages=[{"role": "user", "content": prompt}],
						temperature=config['temperature']
					)
					keywords_json = response.choices[0].message.content.strip().split("\n")
					p.PatientSections[s_n].keywords_json = keywords_json
					p.PatientSections[s_n].keywords = _parse_keywords_from_json_lines(keywords_json)
					print(f"Extracted keywords for section {s}: {p.PatientSections[s_n].keywords}")
				except Exception as e:
					print(f"Error extracting keywords for section {s}: {e}")
					p.PatientSections[s_n].keywords_json = []
					p.PatientSections[s_n].keywords = []
				s_n += 1

	elif type_run == "hpc":
		if llm is None:
			raise ValueError("type_run='hpc' requires a pre-initialized vLLM LLM object passed as `llm`.")

		tokenizer = llm.get_tokenizer()
		section_jobs = []  # list of (s_n, section_key)
		prompts = []
		s_n = 0
		# Per section generate a prompt using the provided template
		#Section_jobs will contain the index of the section and the section name, which will be used to map the outputs back to the correct section in the Patient object
		try:
			for s in sections:
				section_text = sections[s]
				section_text = _truncate_criteria(section_text, tokenizer, criteria_budget)
				if section_text:
					prompt = template.format(
						input_text=section_text, 
						num_keywords=n_keywords
					)
					prompt = _format_to_chat(prompt, llm)
					prompts.append(prompt)
					section_jobs.append((s_n, s))
					s_n += 1

			if not section_jobs:
				return
			structured = StructuredOutputsParams(json=_ANSWER_SCHEMA)
			# Sampling parameters for vLLM generation (takes the parameters to initialize the LLM object)
			sampling_params = SamplingParams(temperature=config['temperature'], 
						max_tokens=max_tokens,structured_outputs=structured)
			try:
				responses = llm.generate(prompts, sampling_params=sampling_params) #for all generated prompts call the llm and pass the parameters
				_extract_keywords_from_outputs(section_jobs, responses, p)

			except Exception as e:
				print(f"Error LLM responses: {e}")
							#Now map the saved section index, name, and the output from the llm back to the Patient object

		except Exception as e:
			print(f"Error during keyword extraction: {e}")
			for s_n, s in section_jobs:
				p.PatientSections[s_n].keywords_json = []
				p.PatientSections[s_n].keywords = []

	else:
		raise ValueError(f"Unknown type_run='{type_run}'. Expected 'debug' or 'hpc'.")
