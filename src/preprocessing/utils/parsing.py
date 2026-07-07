import xml.etree.ElementTree as ET

import re
from pathlib import Path

from dataclasses import dataclass
from typing import Union, List, Optional



# Dataclasses for Clinical Trial representation	
@dataclass(frozen=True)
class Intervention:
    type: str
    name: str
    description: Optional[str]

@dataclass
class ClinicalTrial:
    """Class representing a clinical trial. Most important fields defined as in Nour AiKhoury: Enhancing biomarker based oncology trial matching using large language models"""
    trial_title: str
    name_id: str
    organization_id: str
    brief_summary: str
    trial_status: str
    interventions: List[Intervention]
    minimum_age: Union[float, str]
    maximum_age: Union[float, str]
    gender: str
    accepts_healthy_volunteers: Union[bool, str]
    inclusion_criteria: Optional[str] = None
    exclusion_criteria: Optional[str] = None
    dnf_representation: Optional[str] = None
    

# Parsing functions for Clinical Trial XML files
def nested_items(root:ET, tag:str) -> str:
	'''Direct text: <>text</>
	   '''
	try:
		_found = root.find(tag)
		if _found is None:
			return "EMPTY_" + tag.upper()
		#First direct text
		if _found.text and _found.text.strip():
			return _found.text.strip() # Return the direct text if it exists and is not empty
		all_data = {}
		for c in root.find(tag):
			if c.text and c.text.strip(): # this is the case where the childs have text immediately, we want to return that text
				all_data[c.tag] = c.text.strip()
			# we might have some grandchilds
			for g in c:
				if g.text and g.text.strip(): # this is the case where the grandchilds have text immediately, we want to return that text
					all_data[g.tag] = g.text.strip()
		if len(all_data) == 1:
			return next(v for k,v in all_data.items())
		elif len(all_data) > 1:
			return all_data
		else:
			return "EMPTY_" + tag.upper()
	except Exception as e:
		print(f"Error finding tag '{tag}': {e}")
		return "EMPTY_" + tag.upper()
		
def get_item(root:ET, tag:str) -> str:
	try:
		return getattr(root.find(tag), "text", "EMPTY_" + tag.upper())
	except Exception as e:
		print(f"Error finding tag '{tag}': {e}")
		return "EMPTY_" + tag.upper()
	
def get_conditions(root:ET) -> List[str]:
	# Conditions are defined as a broad spectrum but not necesarily as inclusion or exclusion
	try:
		conditions = []
		for condition in root.findall('condition'):
			if condition.text is not None:
				if "," in condition.text:
					condition = condition.text.split(",")  # some conditions are separated by commas, we want to split them into separate conditions
					conditions.extend([c.strip() for c in condition])
				else:
					conditions.append(condition.text)
		return conditions
	except Exception as e:
		print(f"Error finding conditions: {e}")
		return []
	
def get_interventions(root:ET) -> List[Intervention]:
	try:
		interventions = []
		for inter in root.findall('intervention'):
			intervention = Intervention(
				type=getattr(inter.find('intervention_type'), "text", "EMPTY_INTERVENTION_TYPE"),
				name=getattr(inter.find('intervention_name'), "text", "EMPTY_INTERVENTION_NAME"),
				description=getattr(inter.find('description'), "text", "EMPTY_INTERVENTION_DESCRIPTION")
			)
			interventions.append(intervention)
		return interventions
	
	except Exception as e:
		print(f"Error finding interventions: {e}")
		return []

	except Exception as e:
		print(f"Error finding conditions: {e}")
		return []
	
def get_status(root:ET) -> str:
	try:
		if root.find('overall_status') is not None:
			return getattr(root.find('overall_status'), "text", "EMPTY_OVERALL_STATUS")
		elif root.find('status') is not None:
			return getattr(root.find('status'), "text", "EMPTY_STATUS")
		else:
			return "EMPTY_STATUS"
	except Exception as e:
		print(f"Error finding status: {e}")
		return "EMPTY_STATUS"

def process_inclusion_exclusion(eligibility_criteria: dict) -> dict:
	inclusion_headers = ["inclusion criteria", "inclusive criteria"]
	exclusion_headers = ["exclusion criteria", "exclusive criteria", 
                         "exclusion critieria", "eclusion criteria"]

	def find_first_match(text, headers):
		matches = []
		for header in headers:
			m = re.search(re.escape(header), text, re.IGNORECASE)
			if m:
				matches.append(m.start())
		return min(matches) if matches else None

	eligibility_criteria_copy = eligibility_criteria.copy()  # To avoid modifying the original dictionary
	for k, v in eligibility_criteria.items():
		if "inclusion" in v.lower() or "exclusion" in v.lower():
			inc_pos = find_first_match(v, inclusion_headers)
			exc_pos = find_first_match(v, exclusion_headers)

			if inc_pos is not None and exc_pos is not None:
				eligibility_criteria_copy['criteria'] = (v[inc_pos:exc_pos].strip(), v[exc_pos:].strip())
			elif inc_pos is not None:
				eligibility_criteria_copy['criteria'] = (v[inc_pos:].strip(), None)
			elif exc_pos is not None:
				eligibility_criteria_copy['criteria'] = (None, v[exc_pos:].strip())
			else:
				eligibility_criteria_copy['criteria'] = (None, None)
	return eligibility_criteria_copy

	
def parse_eligibility_criteria(eligibility_criteria):
	'''Parse a dictionary of eligibility criteria childs and grandchilds to extract inclusion and exclusion criteria and other relevant information.'''
	variables = ['gender', 'minimum_age', 'maximum_age', 'healthy_volunteers']
	gender = "EMPTY_GENDER"
	minimum_age = "EMPTY_MINIMUM_AGE"
	maximum_age = "EMPTY_MAXIMUM_AGE"
	healthy_volunteers = "EMPTY_HEALTHY_VOLUNTEERS"
	for v in variables:
		if v in eligibility_criteria:
			exec(f"{v} = eligibility_criteria['{v}']")

	# Now inclusion and exclusion
	processed_eligibility_criteria = process_inclusion_exclusion(eligibility_criteria)
	return processed_eligibility_criteria

def age_to_years(age_str: str, empty_case:str) -> Optional[float]:
	'''Convert age strings like "18 Years", "6 Months", "2 Weeks" to years as a float.'''
	try:
		match = re.match(r'(\d+)\s*(Year[s]?|Month[s]?|Week[s]?|Day[s]?|Hour[s]?)', age_str, re.IGNORECASE)
		if match:
			value, unit = match.groups()
			value = float(value)
			unit = unit.lower()
			if unit == 'years' or unit == 'year':
				return int(value)
			elif unit == 'months' or unit == 'month':
				return int(value / 12)
			elif unit == 'weeks' or unit == 'week':
				return int(value / 52)
			elif unit == 'days' or unit == 'day':
				return int(value / 365)
			elif unit == 'hours' or unit == 'hour':
				return f'{int(value / (365 * 24))}'
		else:
			return empty_case
	except Exception as e:
		return empty_case

def parse_healthy_volunteers(hv_str: str, empty_case:str) -> Optional[bool]:
	'''Convert healthy volunteer strings like "Accepts Healthy Volunteers", "Does Not Accept Healthy Volunteers" to a boolean.'''
	try:
		hv_str = hv_str.lower()
		if "accepts healthy volunteers" in hv_str:
			return True
		elif "does not accept healthy volunteers" in hv_str or "no" in hv_str:
			return False
		else:
			return empty_case
	except Exception as e:
		return empty_case
	


def parse_clinical_trial_xml(xml_path:str) -> ClinicalTrial:
	try:
		tree = ET.parse(xml_path)
		root = tree.getroot()
		official_title = get_item(root, 'official_title')

		file_name = Path(xml_path).name
		organization_id = getattr(root.find('id_info').find('org_study_id'), "text", "EMPTY_ORGANIZATION_ID")

		brief_summary = nested_items(root, 'brief_summary')

		conditions = get_conditions(root)
		status = get_status(root)

		#Intervention is what the trial is doing to the patients
		interventions = get_interventions(root)

		eligibility_criteria = nested_items(root, 'eligibility')
		if isinstance(eligibility_criteria, str) and eligibility_criteria.startswith("EMPTY_"):
			return None #wont append
		eligibility_criteria = parse_eligibility_criteria(eligibility_criteria)

		minimum_age=age_to_years(eligibility_criteria.get('minimum_age', "EMPTY_MINIMUM_AGE"), "EMPTY_MINIMUM_AGE")
		maximum_age=age_to_years(eligibility_criteria.get('maximum_age', "EMPTY_MAXIMUM_AGE"), "EMPTY_MAXIMUM_AGE")
		gender=eligibility_criteria.get('gender', "EMPTY_GENDER")
		healthy_volunteers=parse_healthy_volunteers(eligibility_criteria.get('healthy_volunteers', "EMPTY_HEALTHY_VOLUNTEERS"), "EMPTY_HEALTHY_VOLUNTEERS")
		inclusion=eligibility_criteria.get('criteria', (None, None))[0]
		inclusion = re.sub(r'[ \t]+', r' ', inclusion).strip()
		exclusion=eligibility_criteria.get('criteria', (None, None))[1]
		exclusion = re.sub(r'[ \t]+', r' ', exclusion).strip()
		exclusion = f"!	{exclusion}" if exclusion else None

		return ClinicalTrial(
			trial_title=official_title,
			name_id=file_name,
			organization_id=organization_id,
			brief_summary=brief_summary,
			trial_status=status,
			interventions=interventions,
			minimum_age=minimum_age,
			maximum_age=maximum_age,
			gender=gender,
			accepts_healthy_volunteers=healthy_volunteers,
			inclusion_criteria=inclusion,
			exclusion_criteria=exclusion
		)
	except ET.ParseError as e:
		print(f"Error parsing XML file '{xml_path}': {e}")
		return None
	except Exception as e:
		print(f"Unexpected error processing file '{xml_path}': {e}")
		return None

def process_clinical_trials(directory: str,n:Optional[int]= None, word_filter: Optional[List[str]]=None) -> List[ClinicalTrial]:

	'''Process all clinical trial XML files in a directory and return a list of ClinicalTrial objects.
	:param directory: Directory containing XML files
	:param n: Optional limit on the number of files to process
	:param word_filter: Optional list of words to filter trials by brief summary
	:return: List of ClinicalTrial objects'''
	files = list(Path(directory).glob('*.xml'))
	if not files:
		print(f"No XML files found in directory: {directory}")
		return None
	
	if n is not None:
		files = files[:n]
	
	trials = []
	for file in files:
		trial = parse_clinical_trial_xml(str(file))
		if trial is not None:
			if word_filter is not None:
				if any(word in trial.brief_summary.lower() for word in word_filter):
					trials.append(trial)
			else:
				trials.append(trial)
	return trials
    
		
