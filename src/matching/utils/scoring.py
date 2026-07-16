from typing import Optional
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from matching.utils.answering import PatientTrialSummary, PatientAllTrialSummaries

THRESHOLD_YES = 0.66
THRESHOLD_NO  = 0.34

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


def _eval_dnf_threeval(dnf: str, assignment: dict[str, str]) -> str:
    """Evaluate a DNF using Kleene three-valued logic.

    Each criterion maps to exactly one question, so the per-criterion threshold
    is applied before combining: YES->True, NO->False, N/A->unknown.

    Kleene rules: No dominates AND, Yes dominates OR.
    Implemented via the "all-True vs all-False" equivalence: if fixing all unknowns
    to True gives the same result as fixing them to False, the unknowns are
    irrelevant and the outcome is determinate; otherwise the result is N/A.
    """
    na_qs   = [q for q, v in assignment.items() if v == "N/A"]
    known   = {q: (v == "YES") for q, v in assignment.items() if v != "N/A"}

    if not na_qs:
        return "Yes" if _eval_dnf(dnf, known) else "No"

    # if the N/A questions do not affect the outcome regardless of their value, they are irrelevant
    # Yes or No
    result_all_true  = _eval_dnf(dnf, {**known, **{q: True  for q in na_qs}})
    result_all_false = _eval_dnf(dnf, {**known, **{q: False for q in na_qs}})

    if result_all_true == result_all_false:
        return "Yes" if result_all_true else "No"
    return "N/A"


def score_trial(trial_summary: PatientTrialSummary) -> str:
    """Score a single trial for a patient based on the DNF and question answers.
    Returns 'Yes', 'No', or 'N/A'.

    Since each criterion corresponds to exactly one question, the per-criterion
    probability is trivially 1 (YES), 0 (NO), or 0.5 (N/A). The threshold is
    applied per-criterion first, then criteria are combined via the trial DNF
    using Kleene three-valued logic.
    """
    dnf = trial_summary.trial_DNF
    if not dnf:
        return "N/A"

    assignment = {
        f"Q{i+1}": qa.answer.strip().upper()
        for i, qa in enumerate(trial_summary.question_answers)
    }

    return _eval_dnf_threeval(dnf, assignment)


def score_patient(patient_summaries: PatientAllTrialSummaries) -> dict[str, str]:
    """Score all trials for a patient.
    Returns a dict of trial_id -> 'Yes'/'No'/'N/A'.
    """
    return {
        t.trial_id: score_trial(t)
        for t in patient_summaries.trial_summaries
    }


# ---------------------------------------------------------------------------
# Detailed scoring: numeric score + per-question polarity + reasoning
# ---------------------------------------------------------------------------

def _normalize_dnf(dnf: str) -> str:
    """Strip JSON artifacts and normalize operators to uppercase."""
    expr = dnf.strip()
    if expr.startswith('"') and expr.endswith('",'):
        expr = expr[1:-2].strip()
    elif expr.startswith('"') and expr.endswith('"'):
        expr = expr[1:-1].strip()
    expr = re.sub(r'\bAND\b', 'AND', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bOR\b',  'OR',  expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bNOT\b', 'NOT', expr, flags=re.IGNORECASE)
    return expr


def _parse_dnf_clauses(dnf: str) -> list[dict[str, bool]]:
    """Parse a DNF expression into its OR-clauses.

    Returns a list of dicts, one per clause, mapping Qi -> required_bool
    (True = must be YES, False = must be NO / appears under NOT).
    Returns an empty list if the expression cannot be parsed.
    """
    expr = _normalize_dnf(dnf)

    # Split on top-level OR — clauses may be wrapped in parentheses
    # We split on ' OR ' to avoid matching inside variable names
    clause_strings = re.split(r'\bOR\b', expr) # Because we expect that the LLM gave us the combination of answers that an OR generate (check trialsQuestionsPrompts.py)
    clauses = []
    for cs in clause_strings:
        cs = cs.strip().strip('()')
        literals = re.split(r'\bAND\b', cs)
        clause: dict[str, bool] = {}
        for lit in literals:
            lit = lit.strip().strip('()')
            if lit.startswith('NOT'):
                q = lit[3:].strip()
                clause[q] = False   # requires NO
            else:
                clause[lit] = True  # requires YES
        if clause:
            clauses.append(clause)
    return clauses


def score_trial_detailed(
    trial_summary: PatientTrialSummary,
    na_weight: float = 0.0,
) -> dict:
    """Score a trial with a numeric score and per-question breakdown.

    For each OR-clause in the DNF the fraction of its requirements that the
    patient's answers satisfy is computed.  The clause with the highest score
    (the 'best path to eligibility') is chosen as the reference.

    Args:
        trial_summary: the answered trial for one patient.
        na_weight: contribution of an N/A answer toward a met requirement
            (0.0 = ignored, 0.5 = half credit).  Must be in [0, 1].

    Returns a dict with keys:
        score      – float [0, 1]: fraction of requirements met in best clause
                     (None if no DNF / no clauses parsed).
        eligible   – 'Yes' / 'No' / 'N/A' (Kleene evaluation, same as score_trial).
        questions  – list of per-question dicts:
            question_id  – 'Q1', 'Q2', …
            question     – the question text
            required     – 'YES' | 'NO' | None (if Qi not in the best clause)
            answer       – the patient's answer ('YES' / 'NO' / 'N/A')
            confidence   – model confidence score
            met          – True / False / None (None when answer is N/A and na_weight==0)
            reasoning    – answer_reasoning from the LLM
    """
    dnf = trial_summary.trial_DNF
    eligible = score_trial(trial_summary)   # reuse existing Kleene logic

    if not dnf:
        return {"score": None, "eligible": eligible, "questions": []}

    clauses = _parse_dnf_clauses(dnf)
    if not clauses:
        return {"score": None, "eligible": eligible, "questions": []}

    # Index answers by question key
    answers = {
        f"Q{i+1}": qa
        for i, qa in enumerate(trial_summary.question_answers)
    }

    # Find the clause whose requirements best match the patient's answers
    best_score = -1.0
    best_clause: dict[str, bool] = {}
    for clause in clauses:
        if not clause:
            continue
        points = 0.0
        for q, required_yes in clause.items():
            qa = answers.get(q)
            if qa is None:
                continue
            ans = qa.answer.strip().upper()
            if ans == "N/A":
                points += na_weight
            elif (ans == "YES") == required_yes:
                points += 1.0
        clause_score = points / len(clause)
        if clause_score > best_score:
            best_score = clause_score
            best_clause = clause

    # Build per-question details anchored to the best clause
    question_details = []
    for i, qa in enumerate(trial_summary.question_answers):
        q_key = f"Q{i+1}"
        required_yes = best_clause.get(q_key)   # None if not in this clause
        ans = qa.answer.strip().upper()

        if required_yes is None:
            required = None
            met = None
        else:
            required = "YES" if required_yes else "NO"
            if ans == "N/A":
                met = None if na_weight == 0.0 else bool(na_weight >= 0.5)
            else:
                met = (ans == "YES") == required_yes

        question_details.append({
            "question_id": q_key,
            "question":    qa.question,
            "required":    required,
            "answer":      qa.answer,
            "confidence":  qa.confidence,
            "met":         met,
            "reasoning":   qa.answer_reasoning,
        })

    return {
        "score":    round(best_score, 4) if best_score >= 0 else None,
        "eligible": eligible,
        "questions": question_details,
    }


def score_patient_detailed(
    patient_summaries: PatientAllTrialSummaries,
    na_weight: float = 0.0,
) -> dict[str, dict]:
    """Detailed scoring for all trials of a patient.

    Returns a dict of trial_id -> score_trial_detailed(...) result.
    Pass na_weight > 0 (e.g. 0.5) to give partial credit to N/A answers.
    """
    return {
        t.trial_id: score_trial_detailed(t, na_weight=na_weight)
        for t in patient_summaries.trial_summaries
    }