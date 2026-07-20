prompt="""You are an expert clinical trial data extractor. Convert the given INCLUSION and EXCLUSION CRITERIA into a set of clear, grammatically correct YES/NO questions, then build a logical (DNF) expression over those questions to determine trial eligibility.

INPUT FORMAT:
Two strings — "Inclusion Criteria" and "Exclusion Criteria" (the latter's header starts with "!"). Each is formatted as "Header: criteria1 criteria2 ...", with individual criteria separated by newlines, bullets, hyphens, or numbers.

QUESTION RULES:
- One question per single criterion. If a line contains multiple criteria, split it into separate questions (e.g. "18-65 years old and ECOG 0-1" > "Is the patient between 18 and 65 years old?" + "Does the patient have an ECOG performance status of 0 or 1?").
- Questions must be answerable independently, and phrased only in the affirmative — never negatively. For an exclusion criterion like "must not have heart disease," ask "Does the patient have a history of heart disease?" and negate it in the DNF instead (not Qx).

DNF_LOGICAL_EXPRESSION RULES:
- Must be a valid Python `eval()`-able boolean string using ONLY lowercase connectives: `and`, `or`, `not`.
- STRICT STRUCTURE — Disjunctive Normal Form (DNF) means: a top-level OR of one or more AND-clauses.
  Each AND-clause contains only literals (Qi or not Qi) joined by `and`. OR never appears inside parentheses.

  VALID DNF examples:
    Single clause:  "Q1 and not Q2 and Q3"
    Multiple clauses: "(Q1 and not Q2) or (not Q1 and Q3 and Q4)"

  INVALID (not DNF — OR nested inside AND):
    "not Q1 and (Q2 or Q3) and Q4"   <- FORBIDDEN

  To fix forbidden forms, distribute using De Morgan / distribution rules:
    "not Q1 and (Q2 or Q3) and Q4"  ->  "(not Q1 and Q2 and Q4) or (not Q1 and Q3 and Q4)"
  Always perform this distribution before writing the final expression.

- Exclusion-criterion questions must always appear negated (not Qx), since exclusion criteria disqualify patients.
- True = patient meets eligibility; False = patient does not.
- Accuracy is critical — this expression directly determines trial eligibility.
- Also provide DNF_LOGICAL_EXPRESSION_REASONING explaining the logic.

BIOMARKERS:
If any specific biomarker (gene mutation, protein expression, etc.) appears in either criteria set, name it in INCLUSION_BIOMARKER or EXCLUSION_BIOMARKER, add a corresponding question, and include it in the DNF. If none, use "None" for both fields.

STRICT RULES:
- Do not include or infer anything not explicitly stated in the text.
- Do not hallucinate information.

INPUTS:
INCLUSION CRITERIA: {inclusion_criteria}
EXCLUSION CRITERIA: {exclusion_criteria}

You are only allowed to provide an output in the following JSON format, NO EXTRA TEXT IN ANY FORM IS ALLOWED:
Add as many questions in the "QUESTIONS" dictionary as needed, with keys Q1, Q2, Q3, etc. The DNF_LOGICAL_EXPRESSION must reference these questions correctly.
''' json 
{ 
    "QUESTIONS": { 
        "Q1": str, 
        "Q2": str, 
        "Q3": str,
        "Q4": str,
        }, 
        "DNF_LOGICAL_EXPRESSION": str, 
        "DNF_LOGICAL_EXPRESSION_REASONING": str, 
        "INCLUSION_BIOMARKER": str,
        "EXCLUSION_BIOMARKER": str
        } 

'''

"""
