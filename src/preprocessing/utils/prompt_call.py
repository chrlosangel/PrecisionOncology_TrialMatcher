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


# Properties are key-value pairs 
#Sometimes you want to say that, given a particular kind of property name, the value should
# match a particular schema. That's where patternProperties comes in: it maps 
# regular expressions to schemas.
#additional properties is just to control any other property that is not in the schema

_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
	   "QUESTIONS": {"type": "object"},
        "DNF_LOGICAL_EXPRESSION": {"type": "string"},
	   "DNF_LOGICAL_EXPRESSION_REASONING": {"type": "string"},
	   "INCLUSION_BIOMARKER": {"type": "string"},
	   "EXCLUSION_BIOMARKER": {"type": "string"}
    },
    "required": ["QUESTIONS", "DNF_LOGICAL_EXPRESSION", "DNF_LOGICAL_EXPRESSION_REASONING"]
}

def load_DNF_prompt(template_path: str) -> str:
	'''Load the DNF prompt template from a file.'''
	try:
		with open(template_path) as f:
			raw = f.read()
			raw = raw.replace('{',"{{").replace('}',"}}")

			prompt_template = raw.split('"""')[1]
			prompt_template = prompt_template.replace("{{inclusion_criteria}}", "{inclusion_criteria}").replace("{{exclusion_criteria}}", "{exclusion_criteria}")
			return prompt_template
		
	except FileNotFoundError:
		print(f"Prompt template file not found: {template_path}")
		sys.exit(1)

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

def generate_DNF(trial:ClinicalTrial,template:str ,config: dict,
			  type_run: str = 'debug',llm = None)-> ClinicalTrial:

	'''Generate a DNF representation of the trial's eligibility criteria using a language model.
	:param trial: ClinicalTrial object
	:param template: Prompt template string
	:param config: Configuration dictionary for the language model
	:param type_run: Type of run, either 'debug' which uses a local vLLM OpenAI comparible server (localhost:8000)
	 			 	'hpc' uses a preinitialized vLLM LLM object for a direct batched inference'''

	if type_run not in ['debug', 'hpc']:
		raise ValueError("type_run must be either 'debug' or 'hpc'")

	max_tokens = config.get('max_tokens', 1024)
	max_context = config.get('max_context', 2048)
	# Reserve space for output and template overhead
	criteria_budget = (max_context - max_tokens - 200) // 2

	if type_run == 'debug':
		client = openai.OpenAI(base_url="http://localhost:8000/v1",api_key="unused")

		try:
			tokenizer = llm.get_tokenizer()
			# Truncate inclusion and exclusion criteria to fit within the token budget
			inclusion = _truncate_criteria(trial.inclusion_criteria, tokenizer, criteria_budget)
			exclusion = _truncate_criteria(trial.exclusion_criteria, tokenizer, criteria_budget)
			prompt = template.format(
						inclusion_criteria=inclusion,
						exclusion_criteria=exclusion)
			prompt = _format_to_chat(prompt, llm)

			response = client.chat.completions.create(
				model=config['model'],
				messages=[ {"role": "system", "content": "You are an expert in clinical trial eligibility criteria. Output only valid JSON, no extra text or explanations."},
					{"role": "user", "content": prompt}],
				temperature=config['temperature'],
				max_tokens=max_tokens,
			)
			trial.dnf_representation = response.choices[0].message.content.strip().split("\n")

		except Exception as e:
			print(f"Error generating DNF: {e}")
			
			trial.dnf_representation = None
	elif type_run == 'hpc':
		try:
			if llm is None:
				raise ValueError("type_run='hpc' requires a pre-initialized vLLM LLM object passed as `llm`.")

			tokenizer = llm.get_tokenizer()
			inclusion = _truncate_criteria(trial.inclusion_criteria, tokenizer, criteria_budget)
			exclusion = _truncate_criteria(trial.exclusion_criteria, tokenizer, criteria_budget)
			prompt = template.format(
						inclusion_criteria=inclusion,
						exclusion_criteria=exclusion)
			prompt = _format_to_chat(prompt, llm)
			structured = StructuredOutputsParams(json=_ANSWER_SCHEMA)
			sampling_params = SamplingParams(temperature=config['temperature'], max_tokens=max_tokens, structured_outputs=structured)
               
			response = llm.generate(prompt, sampling_params=sampling_params)
			print(response[0].outputs[0].text.strip().split("\n"))
			trial.dnf_representation =response[0].outputs[0].text.strip().split("\n")
		except Exception as e:
			print(f"Error generating DNF: {e}")
			trial.dnf_representation = None

