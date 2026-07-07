import xml.etree.ElementTree as ET

import json
import re
from pathlib import Path
from enum import Enum

from dataclasses import dataclass
from typing import Union, List, Optional,Tuple
from re import Match
from ast import pattern


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import sys

import requests



def main_(path:str = None,type_analysis: str = "default") -> None:
	#why is the path optional? because we want to be able to run the code without providing a path, in which case we will just do the default analysis without filtering by CiViC
	# Main Function
	# path "Initial path wheter is CiViC or not"
	if path is not None and type_analysis == "CiViC":
		try:
			civic = pd.read_table(path, sep='\t',index_col=False)
			expanded_path=Path(path).parent / "expanded_variants.tsv"
			expanded_df = expand_alias(civic, output=expanded_path)
			processed_path=Path(path).parent / "processed_civic.tsv"
			civic_df = process_civic_data(civic, output=processed_path)
			unique_profiles = unique_genes(civic_df)
			synonyms=gene_symbols_lookup(unique_profiles, output_file=Path(path).parent / "gene_synonyms.tsv")

			return civic, civic_df, expanded_df, synonyms
		
		except FileNotFoundError:
			print(f"File not found: {path}, please check the path and try again.")
			sys.exit(1)
	if path is not None and type_analysis == "default":
		# We don't want to filter by CiViC
		# Default analysis code here
		pass

def expand_alias(civic_df: pd.DataFrame, output: str) -> pd.DataFrame:
	"""
	Expand the gene aliases in the civic dataframe.
	Modified from oncotrialLLM
	"""
	try:
		civic_df = civic_df[['gene','variant','variant_aliases']]
		# some contain numbers so we cannot make them uppercase
		civic_df= civic_df.dropna(subset=['variant_aliases']).reset_index(drop=True) 
		expanded_vars = []
		for index, row in civic_df.iterrows():
			variant = row['variant'].replace('::', '-')
			variant_alias = row['variant_aliases'].split(",")
			variant_alias = [alias.replace('::',"-") for alias in variant_alias]
			for alias in variant_alias:
				expanded_vars.append({'gene': row['gene'], 'variant': row['variant'], 'alias': alias.strip()})
		expanded_df = pd.DataFrame(expanded_vars)
		expanded_df.to_csv(output, index=False, sep='\t')
		return expanded_df
		
	except KeyError as e:
		print(f"Missing expected column in DataFrame: {e}")
		sys.exit(1)
	
def process_civic_data(civic: pd.DataFrame,output: str) -> pd.DataFrame:
	"""
	Process the civic data to extract relevant information for clinical trials.
	"""
	# make all civic columns uppercase no assign
	civic_df = civic.copy() 
	civic_df['gene'] = civic_df['gene'].str.upper()
	civic_df['variant'] = civic_df['variant'].str.upper()
	civic_df['variant_aliases'] = civic_df['variant_aliases'].str.upper()
	civic_df['variant'] = civic_df['variant'].apply(clean_variant_name)

	civic_df.to_csv(output, index=False, sep='\t')
	return civic_df

def clean_variant_name(variant: str) -> str:
	"""Clean and standardize variant names for better matching."""
	variant = re.sub(r'EX(\d+)', r'EXON  \1', variant)
	variant = re.sub(r'\bINS\b', r'INSERTION', variant)
	variant = re.sub(r'\bAND\b\s*', '', variant) #remove "AND" and any following whitespace
	variant = re.sub(r'\bDEL\b', r'DELETION', variant)
	# Handle cases like "EGFRvIII" to "EGFR VIII"
	variant = re.sub(r'([A-Za-z]\d+[A-Za-z])-([A-Za-z]\d+[A-Za-z])', r'\1 \2', variant)

	# Some variants are written as "A149T (c.445G>A)" HGVS notation and we want to extract just "A149T"
	#[^\)]* everything that is not a closing parenthesis, so we can remove the HGVS notation
	# changed to C because we apply uppercase before
	#\)? to capture the optional closing parenthesis at the end of the HGVS notation
	variant = re.sub(r'\s*\(?C\.[^\)]*\)?', '', variant)
	return variant.strip().upper()
	
def unique_genes(civic_df: pd.DataFrame) -> List[str]:
	"""Get unique genes and their counts from the civic dataframe."""
	return list(civic_df['gene'].unique())

def gene_symbols_lookup(gene_symbols: List[str], output_file):
	input_dict = {}
	output_list= []
	url = f"https://mygene.info/v3/query"

	for gene in gene_symbols:
		parameters = {
			'q': f'symbol:{gene}',
			'species': 'human',
			'fields':'symbol,alias,name,entrezgene' #synonyms
		}

		response = requests.get(url, params=parameters)
		data = response.json()
		if data['hits'] :
			input_dict[gene] = {
				"gene" : data['hits'][0]['symbol'],
				"synonyms": list(data['hits'][0].get('alias', [])) + [data['hits'][0].get('entrezgene', '')]
			}
		else:
			print(f"No data found for gene: {gene}")
			continue

	for gene, details in input_dict.items(): 
		for synonym in details['synonyms']:
			output_list.append({'gene': details['gene'], 'symbol': gene, 'synonym': synonym})

	df = pd.DataFrame(output_list)
	df.to_csv(output_file, index=False, sep='\t')
	return df