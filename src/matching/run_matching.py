import argparse
import sys
from pathlib import Path
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matching.utils.scoring as scoring

def argument_parser():
     parser=argparse.ArgumentParser(description="Match patients to clinical trials based on their answers to LLM-generated questions.")
     parser.add_argument("--patients_database_path", required=True, help="Path to the patients database. This is a ChromaDB database. If generated with run_patients.py it should be named chromaDB_patients.")
     parser.add_argument("--clinical_trials_file", required=True, help="Path to the clinical trials file. This is a pickle file containing the processed clinical trials.")
     parser.add_argument("--run_all_patients", action="store_true", help="If set, the script will run matching for all patients in the database. If not set, it will only run for a single patient specified by --patient_id.")
     parser.add_argument("--patient_id", type=str, help="Patient ID to run matching for. Required if --run_all_patients is not set.")
     parser.add_argument("--top_trials", type=int, default=2, help="Only if --patient_id is set, if set, the top k trials for the patient will be returned. Default is 2.")
     return parser

def main():
     parser = argument_parser()
     args = parser.parse_args()

     if not args.run_all_patients and not args.patient_id:
          parser.error("--patient_id is required when --run_all_patients is not set.")

     patients_database_path = Path(args.patients_database_path).resolve()
     clinical_trials_file = Path(args.clinical_trials_file).resolve()
     save_dir = clinical_trials_file.parent

     if clinical_trials_file.suffix != ".pkl":
          raise ValueError(f"Clinical trials file '{clinical_trials_file}' is not a pickle file.")
     if not clinical_trials_file.exists():
          raise FileNotFoundError(f"Clinical trials file '{clinical_trials_file}' does not exist. Please provide a valid path to the processed clinical trials pickle file.")
     if not patients_database_path.exists():
          raise FileNotFoundError(f"Patients database path '{patients_database_path}' does not exist. Please provide a valid path to the ChromaDB database.")

     print("Starting patient-trial matching...")


     new_dir = Path(f"{save_dir}/final_answers/").resolve()


     answers_path = (new_dir / "FinalPatientTrialSummaries.pkl").resolve()

     if not answers_path.exists():
          raise FileNotFoundError(f"Final answers file '{answers_path}' does not exist. Please run run_answering.py before runnning the matching scoring")
     print(f"Final answers file '{answers_path}' exists. Proceeding with matching scoring...")
     try:
          with open(answers_path, "rb") as f:
               patient_trial_summaries = pickle.load(f)
     except Exception as e:
          print(f"Error loading final answers file '{answers_path}': {e}")
          sys.exit(1)

     all_results_dir = Path(f"{new_dir}/scoring_results/").resolve()
     if not all_results_dir.exists():
          all_results_dir.mkdir(parents=True, exist_ok=True)
          print(f"Created directory for all results at '{all_results_dir}'.")

     #Final results file for all patients :D It doesn't matter if we run it multiple times, it will be overwritten each time. This is the file that will be used to get the results for a specific patient.
     all_results_file = (all_results_dir / "AllPatientTrialSummariesScores.pkl").resolve()

     # Now we have the patient_trial_summaries, we can proceed to scoring
     if args.run_all_patients:
          if args.patient_id:
               print(f"Warning: --patient_id ('{args.patient_id}') is ignored when --run_all_patients is set.")
          n_patients = len(patient_trial_summaries)
          print(f"Running matching scoring for all {n_patients} patients...")
          try:
               all_patient_results = {}
               for i, p in enumerate(patient_trial_summaries):
                    print(f"  [{i+1}/{n_patients}] Scoring patient '{p.patient_id}'...")
                    results = scoring.score_patient_detailed(p)
                    all_patient_results[p.patient_id] = results

               with open(all_results_file, "wb") as f:
                    pickle.dump(all_patient_results, f)
               print(f"All patient trial summaries scores saved to '{all_results_file}'.")
               print(f"Re-run this script with --patient_id <PATIENT_ID> to get the results for a specific patient.")
          except Exception as e:
               print(f"Error during processing all patients with trials: {e}")
               sys.exit(1)
     else:
          if not all_results_file.exists():
               raise FileNotFoundError(f"All patient results file '{all_results_file}' does not exist. Please run with --run_all_patients to generate it.")
          print(f"All patient results file '{all_results_file}' exists. Loading it...")
          try:
               with open(all_results_file, "rb") as f:
                    all_patient_results = pickle.load(f)
               print(f"All patient trial summaries scores loaded from '{all_results_file}'.")
               if args.patient_id in all_patient_results:
                    #print(f"Results for patient ID '{args.patient_id}': {all_patient_results[args.patient_id]}")
                    try:
                         # Dictionary
                         if args.top_trials is not None:
                              top_k_results = scoring.return_top_n_trials_for_patient(all_patient_results, args.patient_id, n=args.top_trials)
                    except KeyError:
                         print(f"No results found for patient ID '{args.patient_id}'. Please ensure the patient ID is correct and that the matching scoring has been run for this patient.")
                         sys.exit(1)
               else:
                    print(f"No results found for patient ID '{args.patient_id}'. Please ensure the patient ID is correct and that the matching scoring has been run for this patient.")
                    sys.exit(1)
          except Exception as e:
               print(f"Error loading all patient results file '{all_results_file}': {e}")
               sys.exit(1)

if __name__ == "__main__":
     main()