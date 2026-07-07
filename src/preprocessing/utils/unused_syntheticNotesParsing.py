# Parsing synthetic data 
def parse_trec_topics(xml_path):
	"""Structured as <topics><topic number>"""
	# Tested works with 2021 and 2022 TREC Clinical Trials topics

	tree = ET.parse(xml_path) # gets the structure of the xml file
	root = tree.getroot() # finds the root of the xml file, which is <topics>
	patients = []
	source = Path(xml_path).stem
	for patient in root.findall("topic"):
		patient_id = patient.get("number")
		query = patient.text
		text = query.strip() # removes leading and trailing whitespace
		patients.extend([Patient(unique_id=f"{source}_{patient_id}", patient_id=int(patient_id), description=text)])
	return patients

def parse_sigir_topics(path):
	"""Structured as <TOP><NUM><TITLE> (TITLE is unclosed, so use regex)"""
	with open(path, 'r') as f:
		content = f.read()
	patients = {}
	source = Path(path).stem
	topics = re.findall(r'<TOP>\s*<NUM>(.*?)</NUM>\s*<TITLE>(.*?)\s*</TOP>', content, re.DOTALL)
	for patient_id, query in topics:
		patient_id = patient_id.strip() #removes leading and trailing whitespace
		query = query.strip()
		patients[patient_id] = {"id": patient_id, "query": query, "source": source}
	return patients

