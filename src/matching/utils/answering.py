import sys
import os
import pickle
from pathlib import Path

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
			prompt_template=prompt_template.replace("{{input_text}}", "{input_text}").replace("{{num_keywords}}", "{num_keywords}")
			return prompt_template

	except FileNotFoundError:
		print(f"Error: The file {template_path} was not found.")
		sys.exit(1)

	return prompt_template
