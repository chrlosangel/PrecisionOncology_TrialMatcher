from . import parsingPatients
from typing import Union, List, Optional
import spacy

def chunk_section(section: str, N_sentences: int, tokenizer: Optional[spacy.Language] = None) -> List[str]:
	"""Per section: Generate N chunks, each chunk contains three sentences, each chunk has an overlap with the last and first chunk of the previous and next chunk respectively"""
	#?<= is a positive lookbehind assertion, which means the regex will look for a position in the text that is preceded by a punctuation mark (., !, or ?) followed by one or more whitespace characters. This allows us to split the section into sentences while keeping the punctuation marks attached to the sentences.
	if not section:
		return []
	
	sentences= [sent.text.strip() for sent in tokenizer(section).sents] # we can use the sentence segmentation from the spacy model, which is more accurate than the regex approach
	chunks = []

	for i in range (0, len(sentences), N_sentences-1): # we step by N_sentences-1 to create an overlap of 1 sentence between chunks
		chunk=sentences[i:i+N_sentences] # get the next N sentences
		if chunk: # if the chunk is not empty
			chunks.append("\n".join(chunk)) # join the sentences back into a chunk and add it to the list of chunks
	return chunks


def process_patients(real_patients: List[parsingPatients.Patient], tokenizer: spacy.Language) -> tuple[List[parsingPatients.Patient], dict]:
	"""Processes a list of Patient objects by extracting sections and chunking them.
	:param real_patients: List of Patient objects with unstructured descriptions
	:param tokenizer: Spacy tokenizer for sentence segmentation
	:return: List of Patient objects with extracted sections and chunks, and a dictionary of average chunk counts per patient
	"""
	real_patients_ = real_patients.copy() # we copy the list of patients to avoid modifying the original list
	average_sections = []
	average_chunks_per_patient = []
	# For each patient
	for i, patient in enumerate(real_patients_):
		patient_section_list=[]
		sections = parsingPatients.split_ct_into_sections(patient.description)
		_,_,_, age, gender = parsingPatients.extract_clinical_history_sections(patient, sections)
		real_patients_[i].age = age
		real_patients_[i].gender = gender
		average_sections.append(len(sections)) 
		average_chunks={} 
		for s in sections:
			chunks = chunk_section(sections[s], 3, tokenizer)
			average_chunks[s] = len(chunks)
			sections[s] = chunks

			section_patient_chuks=[]

			seen_chunks = set() # we use a set to keep track of seen chunks for deduplication
			for chunk_text in sections[s]:
				if not chunk_text or not chunk_text.strip(): # if the chunk is empty or contains only whitespace
					continue
				if chunk_text in seen_chunks:
					continue
				seen_chunks.add(chunk_text)
				section_patient_chuks.append(chunk_text)
				
			patient_section_list.extend([parsingPatients.PatientSection(section_name=s,chunks=section_patient_chuks)])
		
		average_chunks_per_patient.append(average_chunks)
		real_patients_[i].PatientSections = patient_section_list

	return real_patients_, average_chunks_per_patient

