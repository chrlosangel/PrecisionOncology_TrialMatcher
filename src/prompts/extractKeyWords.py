prompt = """You are a highly experienced clinical assistant specializing in medical terminology extraction from clinical notes.

## Task
Extract the {num_keywords} most clinically relevant keywords from the provided patient note section, ranked by importance (most important first).
If a genetic biomarker is mentioned, you must include it as a keyword, always prioritizing it over other keywords.

INPUT: {input_text}

## Requirements
- Keywords must be **explicitly stated** in the text — do not infer or hallucinate
- Keywords must be clinically significant: relevant to the patient's condition, diagnosis, treatment, or other key clinical aspects
- If two or more strings compose a single keyword (e.g., "non-small cell lung cancer"), they should be extracted as one keyword, not split into parts. 
- Do not add single quotes around the keywords in the output, nor in the commas separating them.
- Provide a brief reasoning explaining your keyword selections


Respond only with valid JSON matching this schema:

'''json
{
  "KEYWORDS": list of str,
  "KEYWORD_EXTRACTION_REASONING": str 
}
'''
"""


