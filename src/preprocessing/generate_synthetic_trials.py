"""
Generate synthetic clinical trial XMLs from coral patient notes using Claude API.

Usage:
    python generate_synthetic_trials.py \
        --notes_dir data/coral/toy_set \
        --output_dir data/synthetic/trials/coral \
        --api_key $ANTHROPIC_API_KEY
"""

import argparse
import os
from pathlib import Path
import anthropic

SYSTEM_PROMPT = """You are an expert clinical trialist. Given a de-identified oncology clinical note,
generate a realistic synthetic clinical trial in ClinicalTrials.gov XML format that the patient
WOULD be eligible for. The trial must:
- Target the exact cancer type and stage described in the note
- Require the specific biomarkers present (mutations, receptor status, MSI, etc.)
- Have ECOG eligibility that matches the patient's performance status
- Reflect the patient's treatment history (prior lines, current treatment)
- Include realistic inclusion AND exclusion criteria that are specific enough to be useful for eligibility matching

Output ONLY valid XML with no preamble or explanation. Use this structure:
<clinical_study>
  <id_info>
    <org_study_id>SYNTH-XXX</org_study_id>
    <nct_id>NCT_SYNTH_XXX</nct_id>
  </id_info>
  <brief_title>...</brief_title>
  <official_title>...</official_title>
  <sponsors>
    <lead_sponsor>
      <agency>Synthetic Cancer Research Consortium</agency>
      <agency_class>Other</agency_class>
    </lead_sponsor>
  </sponsors>
  <brief_summary><textblock>...</textblock></brief_summary>
  <detailed_description><textblock>...</textblock></detailed_description>
  <overall_status>Recruiting</overall_status>
  <phase>Phase 2</phase>
  <study_type>Interventional</study_type>
  <study_design_info>
    <allocation>Randomized</allocation>
    <intervention_model>Parallel Assignment</intervention_model>
    <primary_purpose>Treatment</primary_purpose>
    <masking>None (Open Label)</masking>
  </study_design_info>
  <primary_outcome>
    <measure>...</measure>
    <time_frame>...</time_frame>
  </primary_outcome>
  <secondary_outcome>
    <measure>...</measure>
    <time_frame>...</time_frame>
  </secondary_outcome>
  <eligibility>
    <criteria>
      <textblock>
Inclusion Criteria:
- ...

Exclusion Criteria:
- ...
      </textblock>
    </criteria>
    <gender>All</gender>
    <minimum_age>18 Years</minimum_age>
    <maximum_age>N/A</maximum_age>
  </eligibility>
  <intervention>
    <intervention_type>Drug</intervention_type>
    <intervention_name>...</intervention_name>
    <description>...</description>
  </intervention>
  <condition>...</condition>
  <keyword>...</keyword>
</clinical_study>"""

USER_TEMPLATE = """Clinical note:
{note}

Generate a synthetic clinical trial XML that this patient would be eligible for.
Use trial ID: SYNTH-{trial_id:03d} and NCT_SYNTH_{trial_id:03d}."""


def generate_trial_xml(note_text: str, trial_id: int, client: anthropic.Anthropic, model: str) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": USER_TEMPLATE.format(note=note_text, trial_id=trial_id)
        }]
    )
    return response.content[0].text.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic clinical trial XMLs from coral notes.")
    parser.add_argument("--notes_dir", type=str, default="data/coral/toy_set",
                        help="Directory containing coral patient note .txt files")
    parser.add_argument("--output_dir", type=str, default="data/synthetic/trials/coral",
                        help="Output directory for generated XML files")
    parser.add_argument("--api_key", type=str, default=None,
                        help="Anthropic API key (defaults to ANTHROPIC_API_KEY env var)")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6",
                        help="Claude model to use")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing XML files")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Provide --api_key or set ANTHROPIC_API_KEY environment variable.")

    client = anthropic.Anthropic(api_key=api_key)

    notes_dir = Path(args.notes_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    note_files = sorted(notes_dir.glob("*.txt"))
    if not note_files:
        print(f"No .txt files found in {notes_dir}")
        return

    print(f"Found {len(note_files)} notes. Generating trials -> {output_dir}")

    for i, note_path in enumerate(note_files, start=1):
        # Derive output filename from note filename: note_01_breast.txt -> synthetic_trial_01_breast.xml
        stem = note_path.stem  # e.g. note_01_breast
        out_name = stem.replace("note_", "synthetic_trial_", 1) + ".xml"
        out_path = output_dir / out_name

        if out_path.exists() and not args.overwrite:
            print(f"[{i}/{len(note_files)}] Skipping {out_name} (already exists, use --overwrite to regenerate)")
            continue

        print(f"[{i}/{len(note_files)}] Generating {out_name} from {note_path.name}...")
        note_text = note_path.read_text()

        try:
            xml_content = generate_trial_xml(note_text, trial_id=i, client=client, model=args.model)
            out_path.write_text(xml_content)
            print(f"    Saved to {out_path}")
        except Exception as e:
            print(f"    ERROR generating trial for {note_path.name}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()

# Example usage:
# python src/preprocessing/generate_synthetic_trials.py \
#    --notes_dir data/coral/toy_set \
#    --output_dir data/synthetic/trials/coral \
#    --api_key $ANTHROPIC_API_KEY