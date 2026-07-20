prompt="""You are an expert clinical trial data extractor. Convert the given EXCLUSION CRITERIA into a set of clear, grammatically correct YES/NO questions.

QUESTION RULES:
- One question per single criterion. If a line contains multiple criteria, split it into separate questions.
- Questions must be phrased in the affirmative only — never negatively. For a criterion like "must not have heart disease," ask "Does the patient have a history of heart disease?" (the negation is handled automatically in the eligibility logic).
- Number questions Q1, Q2, Q3, ... in order.

BIOMARKERS:
If any specific biomarker (gene mutation, protein expression, etc.) appears in the criteria, name it in EXCLUSION_BIOMARKER and add a corresponding question. If none, use "None".

STRICT RULES:
- Do not include or infer anything not explicitly stated in the text.
- Do not hallucinate information.

INPUTS:
EXCLUSION CRITERIA: {exclusion_criteria}

You are only allowed to provide an output in the following JSON format, NO EXTRA TEXT IN ANY FORM IS ALLOWED:
Add as many questions as needed with keys Q1, Q2, Q3, etc.
''' json
{
    "QUESTIONS": {
        "Q1": str,
        "Q2": str,
        "Q3": str
    },
    "EXCLUSION_BIOMARKER": str
}
'''

"""
