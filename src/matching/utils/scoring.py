from typing import Optional
import re
import sys
from pathlib import Path
from dataclasses import dataclass
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from matching.utils.answering import PatientTrialSummary, PatientAllTrialSummaries



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
    # Extract the N/A questions and the known answers
    na_qs   = [q for q, a in assignment.items() if a == "N/A"]
    #Known, evaluate True for Yes and False for No
    known   = {q: (a == "YES") for q, a in assignment.items() if a != "N/A"}

    # If there are no NA questions just evaluate the DNF
    # Return Yes if the DNF evaluates to True, No if it evaluates to False
    if not na_qs:
        return "Yes" if _eval_dnf(dnf, known) else "No" #if true return Yes, if false return No

    # if the N/A questions do not affect the outcome regardless of their value, they are irrelevant
    # Yes or No
    # Fill all NA questions with True and False, and evaluate the DNF in both cases

    result_all_true  = _eval_dnf(dnf, {**known, **{q: True  for q in na_qs}})
    result_all_false = _eval_dnf(dnf, {**known, **{q: False for q in na_qs}})

    # If the results are the same, then the NA questions do not affect the outcome
    # Return Yes to evaluate further, No to evaluate further, and N/A if the results are different
    if result_all_true == result_all_false:
        return "Yes" if result_all_true else "No" # it eval is true return yes
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

    # extract each questions answer, literally is just getting the answers back
    assignment = {
        f"Q{i+1}": qa.answer.strip().upper()  #retrieve the answer field
        for i, qa in enumerate(trial_summary.question_answers) #qa is a PatirentTrialQuestionAnswer object
    } # basically jsut retrieving the answers per question

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
    q_neg,q_pos = 0,0
    for cs in clause_strings:
        cs = cs.strip().strip('()')
        literals = re.split(r'\bAND\b', cs)
        clause: dict[str, bool] = {}
        for lit in literals:
            lit = lit.strip().strip('()')
            if lit.startswith('NOT'):
                q = lit[3:].strip()
                clause[q] = False   # requires NO
                q_neg += 1
            else:
                clause[lit] = True  # requires YES
                q_pos += 1
        if clause:
            clauses.append(clause)
    
    # clauses is a list of dictionaries, where the dictionaries is the question and the value is the needed result
    return clauses,q_pos,q_neg


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
    # The result is yes if the DNF is satisfied by the answers 
    # The result is no if the DNF is not satisfied by the answers
    # The return is NA if the DNF with the NA questions filled in with both True and False gives different results
    eligible = score_trial(trial_summary)   # reuse existing Kleene logic

    if not dnf:
        return {"score": None, "eligible": eligible, "questions": []}

    # Returns what the results should be for each clause, True (if YES) or False  (if NO, not clauses)
    clauses, q_pos, q_neg = _parse_dnf_clauses(dnf) # List of dictionaties
    if not clauses:
        return {"score": None, "eligible": eligible, "questions": []}

    # Make a dictionary where the key is the question ID
    # and the value is the PatientTrialQuestionAnswer (qa)
    answers = {
        f"Q{i+1}": qa # take all the PatientTrialQuestionAnswer objects for a trial!
        for i, qa in enumerate(trial_summary.question_answers)
    }
    # Find the clause whose requirements best match the patient's answers
    best_score = -1.0
    best_clause: dict[str, bool] = {}
    for clause in clauses: # for each dictionary in the list
        if not clause:
            continue
        points = 0.0
        # The question is the key to get the PatientTrialQuestionAnswer object, and the required_yes is the value needed for that question (True/False)
        # for each question/ which is the clause 
        # We are retrieving each question from the dictionary
        conflicting = {}
        for q, required_yes in clause.items(): # question key, needed answer value
            # Get the PatientTrialQuestionAnswer object for that question
            qa = answers.get(q)  # qa has question, answer, confidence, evidence, answer_reasoning, question interpretation
            if qa is None:
                continue
            ans = qa.answer.strip().upper() # answer for a given question
            if ans == "N/A":
                points += na_weight
            # If the answer is equal to the required answer then we add a point to the score from that clause
            elif (ans == "YES") == required_yes:
                points += 1.0
            # Save those that are conflicted (required_yes is True but answer is NO, or required_yes is False but answer is YES) as 0 points
            else:
                conflicting[q] = (ans, required_yes)
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
            # if true then required is YES, if false then required is NO
            required = "YES" if required_yes else "NO"
            if ans == "N/A":
                met = None if na_weight == 0.0 else bool(na_weight >= 0.5)
            else:
                # met is just those that are met or the conflicting 
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
        "score":    round(best_score, 6) if best_score >= 0 else None,
        "eligible": eligible,
        "trial_id": trial_summary.trial_id,
        "questions": question_details,
    }


@dataclass
class PatientTrialResult:
    patient_id: str
    trial: str
    score: Optional[float]
    eligible: str
    questions: list[dict]


def score_patient_detailed(
    patient_summaries: PatientAllTrialSummaries,
    na_weight: float = 0.0,
) -> dict[str, dict]:
    """Detailed scoring for all trials of a patient.

    Returns a dict of trial_id -> score_trial_detailed(...) result.
    Pass na_weight > 0 (e.g. 0.5) to give partial credit to N/A answers.
    """
    trials=[]
    for t in patient_summaries.trial_summaries:
        result = score_trial_detailed(t, na_weight=na_weight)
        # Patient has n PatientTrialResult objects
        trials.append(PatientTrialResult(
            patient_id=patient_summaries.patient_id,
            trial=t.trial_id,
            score=result["score"],
            eligible=result["eligible"],
            questions=result["questions"]
        ))
    return trials

def return_best_trial_for_patient(
        all_patient_results: dict[str, list[PatientTrialResult]],
        patient_id: str):
    """Return the best trial for a patient based on the highest score."""
    all_results = all_patient_results.get(patient_id, [])
    if not all_results:
        return None, None
    best_patient_trial = 0
    for t in all_results:
        current_score = t.score
        if current_score is None:
            continue
        if current_score > best_patient_trial:
            best_patient_trial = current_score
            best_result = t

    print(f"Best trial for patient '{patient_id}': {best_result.trial}\nScore: {best_patient_trial}\n")
    for q in best_result.questions:
        print(f"* Question: {q['question']}\n-Answer based on patient data: {q['answer']}\n-Is it required? (inclusion:YES, exclusion:NO): {q['required']}\n-Patient met criteria: {q['met']}\n-LLM-Reasoning: {q['reasoning']}")


def return_top_n_trials_for_patient(
        all_patient_results: dict[str, list[PatientTrialResult]],
        patient_id: str,
        n: int = 5) -> list[PatientTrialResult]:
    """Return and print the top n trials for a patient ranked by score.

    Args:
        all_patient_results: dict of patient_id -> list[PatientTrialResult].
        patient_id: the patient to query.
        n: how many top trials to show (default 5).

        Trials with score=None (unparseable DNF) are ranked last.
    """
    all_results = all_patient_results.get(patient_id, [])
    if not all_results:
        print(f"No results found for patient '{patient_id}'.")
        return []

    sorted_results = sorted(
        all_results,
        key=lambda t: t.score if t.score is not None else -1.0,
        reverse=True,
    )
    top_n = sorted_results[:n]

    print(f"Top {len(top_n)} trial(s) for patient '{patient_id}':\n")
    for rank, result in enumerate(top_n, start=1):
        print(f"[{rank}] Trial: {result.trial} | Score: {result.score} | Eligible: {result.eligible}")
        for q in result.questions:
            print(f"  * Question: {q['question']}")
            print(f"    - Answer: {q['answer']}")
            req = q['required']
            inc_exc = "Inclusion" if req == "YES" else ("Exclusion" if req == "NO" else "N/A (not in best clause)")
            print(f"    - Inclusion/Exclusion Criteria: {inc_exc}")
            print(f"    - Patient met criteria: {q['met']}")
            print(f"    - LLM-Reasoning: {q['reasoning']}")
        print()

    return top_n

# Next version will include:
# [] backtracking of all evidence
# [] Summary of all criteria that was conflicting
# [] Add a function that allows to return the path to the trials
