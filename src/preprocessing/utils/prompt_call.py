import sys
import re
from pathlib import Path
from typing import Optional
import logging
import openai
import torch
from tqdm import tqdm
from transformers import AutoTokenizer


from . import parsing as parsing_ClinicalTrials

# DataClass
ClinicalTrial=parsing_ClinicalTrials.ClinicalTrial

print("CUDA available:", torch.cuda.is_available())

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

logging.basicConfig(level=logging.INFO)
logging.getLogger("vllm").setLevel(logging.WARNING)


import json

_INCLUSION_SCHEMA = {
    "type": "object",
    "properties": {
        "QUESTIONS":          {"type": "object"},
        "INCLUSION_BIOMARKER": {"type": "string"}
    },
    "required": ["QUESTIONS", "INCLUSION_BIOMARKER"]
}

_EXCLUSION_SCHEMA = {
    "type": "object",
    "properties": {
        "QUESTIONS":          {"type": "object"},
        "EXCLUSION_BIOMARKER": {"type": "string"}
    },
    "required": ["QUESTIONS", "EXCLUSION_BIOMARKER"]
}

def load_prompt(template_path: str, placeholders: list[str]) -> str:
	'''Load a prompt template from a file, restoring the given placeholder names.
	All braces are first escaped, then the listed placeholders are restored as
	real format fields so that str.format() can substitute them later.
	'''
	try:
		with open(template_path) as f:
			raw = f.read()
			raw = raw.replace('{', '{{').replace('}', '}}')
			prompt_template = raw.split('"""')[1]
			for name in placeholders:
				prompt_template = prompt_template.replace(f'{{{{{name}}}}}', f'{{{name}}}')
			return prompt_template
	except FileNotFoundError:
		print(f"Prompt template file not found: {template_path}")
		sys.exit(1)

# Keep old name as an alias so nothing else breaks
def load_DNF_prompt(template_path: str) -> str:
	return load_prompt(template_path, ['inclusion_criteria', 'exclusion_criteria'])

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
		chat_prompt = hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	except Exception:
		# apply_chat_template Converts a list of dictionaries with "role" and "content" keys to a list of token ids. 
		chat_prompt = hf_tokenizer.apply_chat_template(
			messages, tokenize=False, add_generation_prompt=True,
			chat_template=_FALLBACK_TEMPLATE
		)
	return chat_prompt

def _truncate_criteria(text: Optional[str], tokenizer, max_tokens: int) -> str:
	'''Truncate text to fit within max_tokens using the model tokenizer.'''
	if text is None:
		return ""
	tokens = tokenizer.encode(text)
	if len(tokens) > max_tokens:
		tokens = tokens[:max_tokens]
		text = tokenizer.decode(tokens)
	return text

def _build_dnf(inc_questions: dict, exc_questions: dict) -> tuple[dict, str]:
	"""Merge inclusion/exclusion question dicts and build a guaranteed-valid DNF string.

	Inclusion questions keep their keys (Q1, Q2, ...).
	Exclusion questions are renumbered to continue after inclusion (Q_{n+1}, ...).
	DNF = (Q1 and Q2 and ... and not Q_{n+1} and not Q_{n+2} and ...)
	This is always a single AND-clause -> always valid DNF.
	"""
	# Renumber exclusion questions to continue from inclusion
	n_inc = len(inc_questions) #quet the amount of inclusion questions
	exc_renumbered = {} #store the exclusion questions with new keys
	#Sort by the question number x[0] is the key´ like Q1, [1:] is the question number (slice from 1 to the end)
	# enumerate adds another variable to the loop, i, which is the index, and our dictionary has key and value ()
	for i, (_, q_text) in enumerate(sorted(exc_questions.items(), key=lambda x: int(x[0][1:]))):
		#New key, same qyuestion text 
		exc_renumbered[f"Q{n_inc + i + 1}"] = q_text

	all_questions = {**inc_questions, **exc_renumbered}

	inc_part = " and ".join(sorted(inc_questions.keys(),      key=lambda x: int(x[1:])))
	#  all the elements of the list are added a string not before the question. and then we join each of them with an and, the first one just have a not before, which we later 
	# convert to and not with dnf join
	exc_part = " and ".join(f"not {k}" for k in sorted(exc_renumbered.keys(), key=lambda x: int(x[1:])))
	dnf = " and ".join(filter(None, [inc_part, exc_part]))

	return all_questions, dnf


def generate_DNF(trial: ClinicalTrial,
				 inclusion_template: str,
				 exclusion_template: str,
				 config: dict,
				 type_run: str = 'debug',
				 llm=None) -> ClinicalTrial:
	"""Generate questions from inclusion/exclusion criteria via two separate LLM calls,
	then build a guaranteed-valid DNF expression programmatically.

	:param trial: ClinicalTrial object
	:param inclusion_template: Prompt template for inclusion criteria
	:param exclusion_template: Prompt template for exclusion criteria
	:param config: Configuration dictionary (model_name, temperature, max_tokens, max_context)
	:param type_run: 'debug' (local vLLM OpenAI server) or 'hpc' (pre-initialized LLM object)
	:param llm: Required for type_run='hpc'
	"""
	if type_run not in ['debug', 'hpc']:
		raise ValueError("type_run must be either 'debug' or 'hpc'")

	max_tokens  = config.get('max_tokens', 1024)
	max_context = config.get('max_context', 2048)
	criteria_budget = max_context - max_tokens - 200

	try:
		tokenizer = llm.get_tokenizer()
		inclusion = _truncate_criteria(trial.inclusion_criteria, tokenizer, criteria_budget)
		exclusion = _truncate_criteria(trial.exclusion_criteria, tokenizer, criteria_budget)

		inc_prompt = _format_to_chat(inclusion_template.format(inclusion_criteria=inclusion), llm)
		exc_prompt = _format_to_chat(exclusion_template.format(exclusion_criteria=exclusion), llm)

		if type_run == 'debug':
			client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
			system_msg = {"role": "system", "content": "You are an expert in clinical trial eligibility criteria. Output only valid JSON, no extra text or explanations."}

			inc_response = client.chat.completions.create(
				model=config['model_name'],
				messages=[system_msg, {"role": "user", "content": inc_prompt}],
				temperature=config['temperature'], max_tokens=max_tokens,
			).choices[0].message.content.strip()

			exc_response = client.chat.completions.create(
				model=config['model_name'],
				messages=[system_msg, {"role": "user", "content": exc_prompt}],
				temperature=config['temperature'], max_tokens=max_tokens,
			).choices[0].message.content.strip()

		else:  # hpc
			if llm is None:
				raise ValueError("type_run='hpc' requires a pre-initialized vLLM LLM object.")

			inc_structured = StructuredOutputsParams(json=_INCLUSION_SCHEMA)
			exc_structured = StructuredOutputsParams(json=_EXCLUSION_SCHEMA)
			inc_params = SamplingParams(temperature=config['temperature'], max_tokens=max_tokens, structured_outputs=inc_structured)
			exc_params = SamplingParams(temperature=config['temperature'], max_tokens=max_tokens, structured_outputs=exc_structured)

			# Batch both calls in one generate for efficiency
			responses = llm.generate([inc_prompt, exc_prompt], sampling_params=[inc_params, exc_params])
			inc_response = responses[0].outputs[0].text.strip()
			exc_response = responses[1].outputs[0].text.strip()

		inc_data = json.loads(inc_response)
		exc_data = json.loads(exc_response)

		inc_questions = inc_data.get("QUESTIONS", {})
		exc_questions = exc_data.get("QUESTIONS", {})
		inc_biomarker = inc_data.get("INCLUSION_BIOMARKER", "None")
		exc_biomarker = exc_data.get("EXCLUSION_BIOMARKER", "None")

		all_questions, dnf = _build_dnf(inc_questions, exc_questions)

		result = {
			"QUESTIONS":                     all_questions,
			"DNF_LOGICAL_EXPRESSION":        dnf,
			"DNF_LOGICAL_EXPRESSION_REASONING": (
				f"All {len(inc_questions)} inclusion criteria must be met "
				f"and all {len(exc_questions)} exclusion criteria must be absent."
			),
			"INCLUSION_BIOMARKER": inc_biomarker,
			"EXCLUSION_BIOMARKER": exc_biomarker,
		}
		trial.dnf_representation = [json.dumps(result)]
		print(f"  DNF built: {len(all_questions)} questions — {dnf[:80]}{'...' if len(dnf) > 80 else ''}")

	except Exception as e:
		print(f"Error generating DNF for trial {getattr(trial, 'name_id', '?')}: {e}")
		trial.dnf_representation = None

