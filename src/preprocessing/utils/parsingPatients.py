import xml.etree.ElementTree as ET

import json
import re
from pathlib import Path
from enum import Enum

from dataclasses import dataclass
from typing import Union, List, Optional
from re import Match
from ast import pattern





# Classes for representing
class Gender(str, Enum):
    """Enum type class for representing Gender in topics and clinical trials."""

    unknown = "U"
    male = "M"
    female = "F"
    all = "A"

@dataclass
class PatientSection:
    """dataclass containing patient section data."""
    section_name: str
    chunks: List[str]
    keywords_json: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    
@dataclass
class Patient:
    """dataclass containing patient data."""

    unique_id: str
    patient_id: int
    description: str
    gender: Gender = Gender.unknown
    age: Union[int, float, None] = None

    full_sections: Optional[dict] = None
    PatientSections: Optional[List[PatientSection]] = None


# Parsing functions

def clean_and_normalize_clinical_note(text: str) -> str:
	"""Cleans and normalizes clinical note text by removing extra whitespace, normalizing line breaks, and standardizing common abbreviations."""
	# Remove extra whitespace
	text = re.sub(r'\s+', ' ', text).strip()
	# replace multiple periods with a single period
	text = re.sub(r'\.{2,}', '.', text)
	# ensure punctuation is followed by a single space
	# no digits before or after the punctuation to avoid affecting decimal points or numbers
	text = re.sub(r'(?<!\d)\s*([.,!?;:])\s*(?!\d)', r'\1 ', text)  # inside the ([ is group1 ]) so \1 is put anything inside the [ ] and then add a space after it
    # Multiple * or - will be changed to just one * or -
	# Special characters that are not printable will be replaced with a period and a space to maintain sentence structure without introducing noise
	text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '. ', text)

	return text

def parse_raw_clinical_notes(path: str) -> List[Patient]:
	"""Parses raw clinical notes from a directory and returns a list of Patient dataclass instances.
	in: path: str - the directory path containing the clinical note files. Each file should be named with the patient ID and contain the clinical note text.
	out: List[Patient] - a list of Patient dataclass instances with the parsed clinical note data.
	"""
	patients = []
	# It just looks to the path, not subdicts
	for file in Path(path).glob("*.txt"): # We might need to add other file types in the future, but for now we will only process .txt files
		with open(file, 'r') as f:
			text = f.read().strip() # removes leading and trailing whitespace
			text = clean_and_normalize_clinical_note(text)
			patient_id = file.stem
			patients.extend([Patient(unique_id=f"{file.name}_{patient_id}", patient_id=patient_id, description=text)])
	return patients

#Basics

def extract_age(patient: Patient) -> Optional[int]:
	# the age cannot be preceded by hypen, we added this because we noticed some ranges of treatment were being misclassified as ages, for example "treated for 3-5 years" was being misclassified as age 5. So we added a negative lookbehind to ensure that the age is not preceded by a hypen.
	pattern = r"(?<![-])(\d+)[\s-](?:year|month)s?|age[d]?\s+(\d+)|(\d+)\s*(?:y[/.]?o\.?|mo?)\b"
	matches = re.search(pattern, patient.description, re.IGNORECASE)
	if matches:
		return int(next(g for g in matches.groups() if g is not None))
	return None

def extract_gender(patient: Patient) -> Optional[Gender]:
	# -old y.o. yo or y.o. are common ways to indicate age, so we can use that as a signal that the gender will be nearby.
	#\b is a word boundary, so we only match whole words
	# Most cases have the gender after the word "old" so we can add that as a positive signal
	# ?: is a non-capturing group, so we don't have to worry about the order of the words after "old" because we will jsut check the one that matches
	if re.search(r"\b(?:-old|y\.?o\.?)\s+(male|man|boy)\b", patient.description, re.IGNORECASE):
		return Gender.male
	elif re.search(r"\b(?:-old|y\.?o\.?)\s+(female|woman|girl)\b", patient.description, re.IGNORECASE):
		return Gender.female
	else:
		return Gender.unknown

def extract_past_medical_history(patient_description: str) -> Optional[Match]:
    """Tries to extract a sentence from the patient description that corresponds to
    past medical history. This is a very basic implementation that looks for specific keywords and patterns in text.
    Later steps will involve more complex NLP techniques to extract this information more accurately and robustly.
    :param patient_description: unstructured patient description without specific
        sections
    :return: re.Match object or None if didn't find anything
    
    """
    match = list(re.finditer(
            r"[!\.](?![^!\.]*family)[^!\.]* past medical history.*?\.", 
            patient_description, 
            re.IGNORECASE
        ) or re.finditer(
            r"[!\.](?![^!\.]*family)[^!\.]*has (no )?(a )?(positive )?history.*?\.",
            patient_description,
            re.IGNORECASE,
        ) 
	)
    if match:
        return match
    else:
        return None

def extract_family_history(patient_description: str) -> Optional[Match]:
    """Extracts a sentence with family history using regex match

    :param patient_description: unstructured patient description without specific
            sections
    :return: re.Match object or None if didn't find anything
    """
    #[^\.\n]*\. we changed to this because the use of [^\.]* meant any character except a period, which
    # still allowed for newlines, which caused the regex to match across multiple sencences. So it was finding the first period
    # and then matching everithing until finding the period after the familiy history.
    #
    
    matches=list(re.finditer(
         	r"\.[^\.\n]*family history[^\.\n]*\.", 
            patient_description, 
            re.IGNORECASE
        ) or re.finditer(
            r"\.[^\.\n]*family.*?\.", 
			patient_description,
            re.IGNORECASE)
	)
    if matches:
        return matches
    else:
        return None
    
def split_ct_into_sections(patient_description: str) -> dict:
	"""Split a clinical note into sections, preserving ALL text.

	Longer/more-specific alternatives come first so that e.g.
	'Social History Main Topics' is matched before 'Social History',
	preventing the prefix from being swallowed.  Duplicate section
	headers (an EHR export artifact) are concatenated rather than
	overwritten so no content is lost.  Sections not in this list
	remain part of the preceding section's text.
	"""
	#Split the section if it finds one of these, if the info in the note is not in one of these it will be kept on the preciding section text
	section_pattern = (
		r"("
		# --- Social History sub-sections (must precede generic Social History) ---
		r"Social History Main Topics"
		r"|Social History Narrative"
		r"|Occupational History"
		r"|Other Topics Concern"
		# --- Core history sections ---
		r"|Past(?:\s+(?:Family(?:\s+and\s+)?)?)?Medical History"
		r"|Past Surgical History|Prior Surger(?:y|ies)"
		r"|(?:Past,?\s+)?Family(?:\s+and\s+Social)?\s+History"
		r"|Social History"
		# --- Medications (many EHR label variants) ---
		r"|(?:Current\s+)?Outpatient\s+Encounter\s+Prescriptions?"
		r"|(?:Current\s+)?Facility-Administered\s+Encounter\s+Medications?"
		r"|Current\s+Outpatient\s+(?:Prescriptions?|Medications?)"
		r"|Current(?:\s+\w+){0,4}\s*Medications?"
		# --- Other major sections ---
		r"|(?:Gyne(?:cologic)?(?:\s+History)?|Obstetric\s+History|Reproductive\s+History|Gynecologic History)"
		r"|Allergies?(?:\s*/\s*Contraindications?)?"
		r"|Review\s+of\s+Systems?"
		r"|Physical\s+Exam(?:ination)?"
		r"|Assessment[\s/]+(?:and\s+)?(?:Plan|Recommendations?)"
		r"|Assessment\s+and\s+Recommendations?"
		r"|Lab(?:oratory)?\s+Results?"
		r"|Studies"
		r"|Imaging"
		r"|Vital\s+Signs?"
		r")"
		r"\s*:?\s*"
	)

	# Split the patient description into sections based on the section pattern
	parts = re.split(section_pattern, patient_description, flags=re.IGNORECASE)
	# Parts is Header/nContent thats why on range we step by 2

	# Initialize the sections dictionary with the preamble, everything before the first section header
	sections: dict = {"_preamble": parts[0].strip()}

	# Iterate through the parts and populate the sections dictionary (start from 1 to skip the preamble, step by 2 to get section name and text)
	for i in range(1, len(parts) - 1, 2):
		section_name = parts[i].strip().lower()
		section_text = parts[i + 1].strip()
		if section_name in sections and sections[section_name]: # if we already have this section name and it has text, we concatenate the new text to the existing text
			# Concatenate: handles EHR duplicate headers without dropping text
			sections[section_name] = sections[section_name] + " " + section_text
		else:
			sections[section_name] = section_text

	return sections

def extract_clinical_history_sections(patient: Patient,sections: dict) :
	"""Extracts all clinical history sections from the patient description.
	:param patient: Patient object with unstructured description containing clinical history sections
	:return: tuple of (current medical history text, past medical history text, family medical history text, age) where the medical history texts are strings and age is an integer or None
	"""
	age: Optional[int] = extract_age(patient)
	gender: Optional[str] = extract_gender(patient)

	past_mh_text: Optional[str] = sections.get("past medical history")
	family_mh_text: Optional[str] = sections.get("family history")
	current_mh_text = sections.get("_preamble")

	if not past_mh_text and not family_mh_text: # if we didn't find any sections, we can try to extract them using regex

		pmh: Optional[List[Match]] = extract_past_medical_history(patient.description)
		fmh: Optional[List[Match]] = extract_family_history(patient.description)

			# Because we can have multiple matches for both past medical history and family history we have to decide how to handle that
   		 # can only concatenate NonType if both are not None, so we have to check for that
		if pmh is None:
			pmh = []
		if fmh is None:
			fmh = []
			
		all_matches=sorted(pmh + fmh, key=lambda m: m.start()) 
		# Now remove all spans
		current_mh_text = patient.description
		for m in reversed(all_matches): # we reverse the list of matches so we can remove the spans without affecting the positions of the other matches
			if m:
				current_mh_text = current_mh_text[:m.start()] + current_mh_text[m.end():]
	
		past_mh_text = " ".join(
    	        patient.description[m.start():m.end()] for m in pmh if m
		)
		family_mh_text = " ".join(
				patient.description[m.start():m.end()] for m in fmh if m
		)
	return current_mh_text, past_mh_text, family_mh_text, age, gender
	