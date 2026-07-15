from itertools import product
from typing import Optional
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from matching.utils.answering import PatientTrialSummary, PatientAllTrialSummaries

THRESHOLD_YES = 0.66
THRESHOLD_NO  = 0.34

# DRAFT: This is a draft implementation of scoring logic. It may not be fully correct or complete. Use with caution.

def _eval_dnf(dnf: str, assignment: dict[str, bool]) -> bool:
    """Evaluate a DNF boolean expression given a mapping of Qi -> True/False.
    Handles both uppercase (AND/OR/NOT) and lowercase (and/or/not) operators.
    """
    expr = dnf.strip()
    if expr.startswith('"') and expr.endswith('",'):
        expr = expr[1:-2].strip()
    # Normalize operators to Python lowercase
    expr = re.sub(r'\bAND\b', 'and', expr)
    expr = re.sub(r'\bOR\b',  'or',  expr)
    expr = re.sub(r'\bNOT\b', 'not', expr)
    # Substitute Qi values — sort descending by length to avoid Q1 matching inside Q10
    for q, val in sorted(assignment.items(), key=lambda x: -len(x[0])):
        expr = expr.replace(q, str(val))
    try:
        return bool(eval(expr))
    except Exception:
        return False


def score_trial(trial_summary: PatientTrialSummary) -> str:
    """Score a single trial for a patient based on the DNF and question answers.
    Returns 'Yes', 'No', or 'N/A'.
    """
    dnf = trial_summary.trial_DNF
    if not dnf:
        return "N/A"

    # Build Qi -> answer mapping from ordered question_answers
    raw_answers = {
        f"Q{i+1}": qa.answer.strip().upper()
        for i, qa in enumerate(trial_summary.question_answers)
    }

    na_questions    = [q for q, a in raw_answers.items() if a == "N/A"]
    #If a is YES then set to True, if a is NO then set to False
    known_questions = {q: (a == "YES") for q, a in raw_answers.items() if a != "N/A"}

    if not na_questions:
        result = _eval_dnf(dnf, known_questions)
        return "Yes" if result else "No"

    # Marginalize over all 2^N combinations of N/A answers
    hits = 0
    total = 0
    for combo in product([True, False], repeat=len(na_questions)):
        assignment = {**known_questions, **dict(zip(na_questions, combo))}
        hits  += int(_eval_dnf(dnf, assignment))
        total += 1

    p = hits / total
    if p > THRESHOLD_YES:
        return "Yes"
    elif p < THRESHOLD_NO:
        return "No"
    else:
        return "N/A"


def score_patient(patient_summaries: PatientAllTrialSummaries) -> dict[str, str]:
    """Score all trials for a patient.
    Returns a dict of trial_id -> 'Yes'/'No'/'N/A'.
    """
    return {
        t.trial_id: score_trial(t)
        for t in patient_summaries.trial_summaries
    }