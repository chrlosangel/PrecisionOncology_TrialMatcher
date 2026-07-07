# Building an LLM-Based Patient–Clinical Trial Matching Pipeline: A Comprehensive Step-by-Step Guide

*Synthesized from: Jin et al. (2024, Nature Comms — TrialGPT), Chen et al. (2025, JCO CCI — Scoping Review), Morrison et al. (2025 — Systematic Review), Gueguen et al. (2025, npj Precision Oncology — Prospective Evaluation), Alkhoury et al. (2025, npj Digital Medicine — Biomarker Extraction)*

---

## Overview of the Pipeline

Most LLM-based trial-matching systems share four stages: (1) data acquisition, (2) data preprocessing, (3) retrieval of relevant candidates, and (4) matching and ranking. Below is a detailed breakdown of each stage — what it involves, what tools and data sources are used in the literature, and what techniques are available.

---

## 1. DATA ACQUISITION

### 1.1 Patient Data

| Data Type | Description | Where to Get It | Used By |
|---|---|---|---|
| **Synthetic patient case descriptions** | Short admission-note-style vignettes written by clinicians. Most commonly used in benchmarking. | TREC 2021 CT (http://www.trec-cds.org/2021.html), TREC 2022 CT (http://www.trec-cds.org/2022.html), TREC 2023 CT (http://www.trec-cds.org/2023.html), SIGIR 2016 (https://data.csiro.au/collection/csiro:17152) | TrialGPT, Trial-LLAMA, multiple TREC participants |
| **n2c2 2018 cohort** | Real de-identified clinical notes from diabetic patients with 13 predefined inclusion criteria; includes longitudinal records. | National NLP Clinical Challenges (https://n2c2.dbmi.hms.harvard.edu/) | Wornow et al., Beattie et al., Shi et al., Woo et al. |
| **Real EHR / clinical notes** | Progress notes, discharge summaries, pathology reports, radiology reports, H&P notes. Richest data but hardest to access due to PHI constraints. | Institutional IRB-approved access; MIMIC-III/IV (https://physionet.org/) for de-identified ICU data | Wong et al., Unlu et al. (COPILOT-HF), Lai et al. (pancreatic), Gupta et al. (PRISM) |
| **Case reports / vignettes** | Manually created or adapted from published case reports. | Published medical literature; can be created by domain experts | Devi et al., Ferber et al. |
| **Structured questionnaire answers** | Patient responses to eligibility-based questionnaires, sometimes LLM-generated. | Custom creation; TREC 2023 uses structured Q&A format | Zihang et al., Woo et al. |
| **Claims / prescription data** | Longitudinal diagnoses, procedures, medications from insurance claims. | Institutional data warehouses; synthetic versions via Synthea (https://synthetichealth.github.io/synthea/) | Yuan et al. (stroke trials) |
| **Genomic / molecular profiling** | Somatic mutations, CNVs, fusions, TMB from tumor sequencing. Essential for precision oncology matching. | Institutional molecular tumor boards; AACR GENIE (https://www.aacr.org/professionals/research/aacr-project-genie/) for public cohort data | Gueguen et al. (ProfiLER), Alkhoury et al. (CIViC biomarkers) |

**Key considerations:**
- Most benchmark studies use **synthetic data** (TREC/SIGIR), but prospective evaluations like Gueguen et al. show significant performance gaps between synthetic and real-world scenarios.
- Privacy constraints often necessitate **on-premises LLM deployment** or **synthetic data generation** (e.g., using GPT-4 to create synthetic patient-trial pairs for training smaller models, as done by Zhuang et al. and Woo et al.).
- Patient sample sizes in the literature range from 10 to 1,891 patients, with most studies analyzing fewer than 200.

### 1.2 Clinical Trial Data

| Source | Description | Access |
|---|---|---|
| **ClinicalTrials.gov** | The primary public registry; contains ~470K+ trial records with structured fields (title, conditions, interventions, eligibility criteria, status, sites). | Free API: https://clinicaltrials.gov/data-api/api |
| **Hospital / institutional databases** | Internal trial management systems with real-time recruitment status, often more accurate than ClinicalTrials.gov. | IRB/institution-specific access |
| **International registries** | ChiCTR (China), EU Clinical Trials Register, ISRCTN. | Varies by registry |
| **Pre-processed trial datasets** | Curated subsets for benchmarking. | TREC CT tracks provide pooled judgments for ~26K–27K trials; SIGIR 2016 covers 3,621 trials |

**Key considerations:**
- ClinicalTrials.gov status can be **outdated** — Gueguen et al. found that only 80.5% of trials listed as "recruiting" on ClinicalTrials.gov were actually recruiting at their site.
- The number of active trials in a single country can reach ~23,000 (TrialGPT), so scalability is critical.
- For biomarker-focused matching, Alkhoury et al. used the **CIViC database** (https://civicdb.org) to curate 500 cancer-related genomic biomarkers and queried ClinicalTrials.gov against them, identifying 296 relevant trials.

### 1.3 Biomarker & Knowledge Databases

| Database | Use Case |
|---|---|
| **CIViC** (https://civicdb.org) | Open-source knowledgebase of cancer biomarkers — used to identify actionable variants and link them to trials |
| **AACR GENIE** (https://www.aacr.org/professionals/research/aacr-project-genie/) | Large-scale real-world cancer genomics cohort (~172K patients) — useful for estimating biomarker prevalence |
| **KEGG Pathway Database** | Used by DigitalECMT to interpret molecular alterations and link to trial drug targets |
| **UMLS / SNOMED CT / ICD / LOINC / RxNorm** | Standard medical vocabularies for concept normalization |
| **OMOP CDM** | Common data model for harmonizing EHR data across institutions; used by Criteria2Query 3.0 |

---

## 2. DATA PREPROCESSING

### 2.1 Eligibility Criteria Processing

This module transforms raw trial criteria text into formats suitable for matching. Three main techniques:

#### 2.1.1 Eligibility Criteria Parsing (Structuring)

**What:** Extract key entities (diseases, biomarkers, lab values, medications) and logical relationships from free-text criteria.

**How:**
- **LLM-based extraction:** Use GPT-4 or domain-specific LLMs to identify clinical concepts. Criteria2Query 3.0 achieved precision 0.862, recall 0.922, F1 0.891 for medical concept extraction.
- **Rule-based + NLP hybrid:** Wong et al. reported F1 of 0.648 for histological criteria and 0.725 for biomarker extraction using GPT-4.
- **AutoCriteria** (Datta et al.): An LLM-powered system that extracts detailed eligibility criteria across diverse diseases without manual annotations.

**Tools used in literature:** GPT-4, GPT-3.5, domain-specific NER models, spaCy with biomedical models.

#### 2.1.2 Medical Concept Normalization

**What:** Map extracted entities to standardized vocabularies so that "high blood pressure" matches "hypertension" and "Herceptin" matches "trastuzumab."

**How:**
- Map to **ICD codes** (diseases), **LOINC** (lab tests), **RxNorm** (medications)
- Use **UMLS** as a reference ontology for normalization algorithms
- Some studies map to **OMOP vocabulary** to enable SQL queries against EHR databases in the OMOP CDM
- For genomic biomarkers, normalize gene names to **HGNC symbols** and variant nomenclature to **HGVS** format

**Tools used:** UMLS API, MetaMap, QuickUMLS, SciSpacy with UMLS linker, custom normalization pipelines.

#### 2.1.3 Criteria Rephrasing

**What:** Convert complex eligibility criteria into simplified, independent units while preserving semantics. This is critical for criterion-level matching approaches.

**How:**
- **Decompose into atomic criteria:** Break compound criteria (e.g., "Patient must have EGFR mutation AND be treatment-naive") into individual assessable units.
- **Convert to questions:** PRISM (Gupta et al.) reformulates criteria as yes/no questions, achieving 97% accuracy for question generation and 89% for Boolean logic.
- **Add external definitions:** Shi et al. augment criteria with knowledge from medical ontologies for disambiguation.
- **Structure in Disjunctive Normal Form (DNF):** Alkhoury et al. represent biomarker logic as OR-of-ANDs in JSON format, capturing complex cohort-specific requirements (e.g., "Cohort A requires EGFR L858R OR Cohort B requires KRAS G12C").

**Tools used:** GPT-4 for rephrasing, LLMs with chain-of-thought prompting, Hermes-2-Pro-Mistral-7B (Alkhoury et al. showed this 7B open-source model outperformed GPT-4 at biomarker structuring with F2=0.94 vs 0.29 for inclusion biomarkers in DNF).

### 2.2 Patient Data Processing

#### 2.2.1 Information Extraction

**What:** Extract clinically relevant entities from unstructured patient records.

**How:**
- **NER for clinical entities:** Extract diagnoses, medications, procedures, lab values, genomic findings.
- **LLM-based extraction:** Use GPT-4 or open-source LLMs to identify and extract key patient characteristics from notes.
- **Negation detection:** Ensure "no history of diabetes" is not interpreted as having diabetes. Critical for accurate matching.
- **Temporal reasoning:** Identify when events occurred (e.g., "completed chemotherapy 6 months ago" vs. current treatment).

**Tools used:** MedSpacy, SciSpacy, GPT-4, clinical NER models, regular expressions for lab values.

#### 2.2.2 Normalization & Expansion

**What:** Standardize extracted terms and expand them with related concepts.

**How:**
- **Synonym expansion:** Using SNOMED CT taxonomic structure to expand extracted terms (e.g., "NSCLC" → also search "non-small cell lung cancer," "pulmonary adenocarcinoma").
- **Code mapping:** Map diagnoses to ICD-10, medications to RxNorm, labs to LOINC.
- **Biomarker normalization:** Normalize gene variant names (e.g., "BRAF V600E" = "BRAF p.V600E" = "BRAF c.1799T>A"). Gueguen et al. found that gene variant mismatches (e.g., KRAS G12C vs G12V) were the **single most frequent cause of false positives** (28–51% of errors across tools).

#### 2.2.3 Chunking

**What:** Segment lengthy clinical notes into manageable pieces that fit within LLM context windows.

**How:**
- **Fixed-size chunking:** Split notes into segments of fixed token length (most common approach).
- **Semantic chunking:** Use LLMs to split at natural boundaries while preserving clinical coherence. PRISM explored this, achieving 63.35% accuracy but with high computational costs.
- **Note-type selection:** Select only the most relevant note types (e.g., H&P, oncology notes, pathology reports) rather than processing all notes.

#### 2.2.4 Patient Summary Generation

**What:** Create concise patient profiles from full EHR data.

**How:**
- Use LLMs to generate structured patient summaries from unstructured notes.
- **Keyword extraction:** TrialGPT uses GPT-4 to generate up to 32 keywords ranked by importance from the patient summary.
- **Query rephrasing:** Convert keywords into search queries for retrieval (Peikos et al. used ChatGPT for query generation).

---

## 3. RETRIEVAL

### Purpose

Retrieval acts as a scalable pre-filtering step to narrow down from tens of thousands of candidate trials to a manageable shortlist. This is analogous to the "retrieval" component in RAG (Retrieval-Augmented Generation) architectures.

### 3.1 Retrieval Approaches

| Approach | Description | Used By |
|---|---|---|
| **Lexical (BM25)** | Traditional term-frequency-based matching; captures exact keyword overlap | TrialGPT (combined), Jullien et al., multiple TREC participants |
| **Semantic (Dense retrieval)** | Encode patient/trial data as dense vectors and compute similarity. Captures conceptual alignment even without keyword overlap. | TrialGPT (MedCPT), Ferber et al. (OpenAI Ada embeddings), Chowdhury et al. (LLaMA-2 embeddings) |
| **Hybrid (Lexical + Semantic)** | Combine BM25 and dense retrieval, usually via reciprocal rank fusion. Best overall performance in most studies. | TrialGPT (BM25 + MedCPT with reciprocal rank fusion) |
| **Cross-encoder re-ranking** | Use a cross-encoder model (e.g., BioLinkBERT) to re-rank initial retrieval results with cross-attention between query and document. | Kusa et al. (TCRR with BioBERT), Zhuang et al. |
| **LLM-based re-ranking** | Use GPT-4 or similar to re-rank candidate trials after initial retrieval. | Rybinski et al. (GPT-3.5/GPT-4o re-ranking), Gueguen et al. (TrialGPT + Qwen2.5-7B re-ranking improved NDCG@3 from 0.61 to 0.64) |

### 3.2 TrialGPT-Retrieval (Detailed)

The most thoroughly described retrieval system in the reviewed papers:

1. **Keyword generation:** GPT-4 generates up to 32 keywords from the patient summary, ranked by importance.
2. **Per-keyword retrieval:** Each keyword is sent to both BM25 (lexical) and MedCPT (semantic) retrievers.
3. **Reciprocal rank fusion:** For each keyword, ranks from BM25 and MedCPT are fused. Across keywords, a decaying weight (1/i for keyword rank i) combines scores.
4. **Scoring formula:**
   ```
   score_j = Σ_Ret Σ_{i=1}^{K} 1/(i × (Rank(Ret, w_i, t_j) + C))
   ```
   where C=20 is the reciprocal rank fusion constant, Ret ∈ {BM25, MedCPT}.
5. **Result:** The top-ranked trials form the candidate set. To recall ≥90% of relevant trials, only ~5.5% of the collection needs to be retained (GPT-4 keywords) vs. 7.0% (GPT-3.5 keywords).

### 3.3 Embedding Models for Retrieval

| Model | Type | Used By |
|---|---|---|
| **MedCPT** | Biomedical dense retriever trained on PubMed search logs | TrialGPT |
| **OpenAI Ada embeddings** | General-purpose dense embeddings | 3 studies in Chen et al. review |
| **BERT-based models** (BioBERT, PubMedBERT, SapBERT) | Biomedical domain encoders | Multiple TREC participants, Cerami et al. (MatchMiner-AI) |
| **LLaMA-2 embeddings** | Open-source LLM used as embedding backbone | Chowdhury et al. (Siamese network) |
| **FAISS** | Similarity search library for efficient nearest-neighbor lookup | Used with dense embeddings for scalable retrieval |

### 3.4 Retrieval Performance Metrics

- **Recall@k**: Fraction of relevant trials found in top k results. TrialGPT achieves Recall@500 of 86.2% with GPT-4 keywords.
- **NDCG@1000**: Normalized discounted cumulative gain over deep rankings.
- **Reciprocal Rank (RR)**: Position of the first relevant result.

---

## 4. MATCHING & RANKING

### 4.1 Matching Paradigms

Four main architectures identified across the literature:

#### 4.1.1 Direct LLM-Based Matching (Trial-Level)

**What:** Feed the LLM the full set of eligibility criteria + patient data and ask for a single eligibility decision.

**How:** Prompt an LLM with both inputs and request a classification (Eligible / Not Eligible / Unknown).

**Pros:** Simple, fast per trial.  
**Cons:** Black-box, no criterion-level explainability, may miss nuances.

**Used by:** Wornow et al. (zero-shot GPT-4, Micro-F1=0.93), Devi et al. (GPT-4, accuracy up to 100% on small sets), Lai et al. (GPT-4o, 96.7% accuracy on binary criteria).

#### 4.1.2 Decomposed Criteria Matching (Criterion-Level → Aggregation)

**What:** Decompose criteria into individual units, evaluate each patient-criterion pair separately, then aggregate results to trial-level scores.

**How:**
1. For each criterion, the LLM generates: (a) relevance explanation, (b) relevant sentence locations in patient note, (c) eligibility classification.
2. Eligibility labels per criterion:
   - Inclusion: {Included, Not included, Not enough information, Not applicable}
   - Exclusion: {Excluded, Not excluded, Not enough information, Not applicable}
3. Aggregate criterion-level predictions to trial-level scores (see Ranking below).

**Pros:** Explainable, allows human oversight at criterion level, higher accuracy.  
**Cons:** More API calls, higher cost, slower.

**Used by:** TrialGPT (87.3% criterion-level accuracy, close to expert performance of 88.7–90.0%), PRISM (Gupta et al.), Nievas et al. (Trial-LLAMA).

#### 4.1.3 Embedding-Based Matching

**What:** Generate separate embeddings for patient data and trial criteria, then compute similarity.

**How:** Use BERT-based or LLM-based embeddings + neural networks (e.g., Siamese networks, CNNs) for classification.

**Used by:** Chowdhury et al. (LLaMA-2 + Siamese network, F1=0.92), Yuan et al. (BERT embeddings + CNN), Cerami et al. (MatchMiner-AI with fine-tuned RoBERTa-Large).

#### 4.1.4 Keyword-Based Matching

**What:** Extract keywords from both criteria and patient data, then perform exact or fuzzy matching.

**Used by:** Wong et al. (structured EHR + keyword matching), Rahmanian et al.

### 4.2 Ranking (Trial-Level Score Aggregation)

After criterion-level matching, scores must be aggregated to rank trials. TrialGPT describes two approaches:

#### 4.2.1 Linear Aggregation

Six scores computed from criterion-level predictions:
- % inclusion criteria predicted as "included"
- % inclusion criteria predicted as "not included"
- % inclusion criteria with "not enough information"
- % exclusion criteria predicted as "excluded"
- % exclusion criteria predicted as "not excluded"
- % exclusion criteria with "not enough information"

Criteria labeled "not applicable" are excluded from denominators.

#### 4.2.2 LLM Aggregation

Use an LLM to generate two scores from criterion-level predictions:
- **General relevance score** (0–100): How relevant is the patient to the trial?
- **Eligibility score** (-100 to +100): How eligible is the patient? Negative = ineligible, positive = eligible.

Constraint: |Eligibility| ≤ Relevance (eligibility cannot exceed relevance).

#### 4.2.3 Feature Combination (Best Performance)

Combine linear and LLM aggregation:
```
combination = %met_inclusion - I(%unmet_inclusion > 0)
              - I(%met_exclusion > 0) + %LLM_relevance
              + %LLM_eligibility
```
where I() is an indicator function.

**Performance:** GPT-4-based TrialGPT with feature combination achieved NDCG@10=0.7275, P@10=0.6688, AUROC=0.7979 — about 43.8% better than the best baseline (BioLinkBERT trained on MedNLI).

### 4.3 Exclusion of Ineligible Trials

Modeled as binary classification. Key signals:
- Any exclusion criterion predicted as "excluded" → strong signal for ineligibility
- Any inclusion criterion predicted as "not included" → strong signal for ineligibility
- In most eligible patient-trial pairs, no exclusion criterion is labeled "excluded" and no inclusion criterion is labeled "not included."

### 4.4 Explanation Generation

A distinguishing feature of LLM-based approaches:
- **Criterion-level explanations:** TrialGPT generates natural language rationales for each patient-criterion pair. 87.8% rated "correct" by expert evaluators.
- **Evidence sentences:** TrialGPT identifies relevant sentences in the patient note for each criterion. Precision=90.1%, Recall=87.9%, F1=88.6% — comparable to human experts (86.9–91.5%).
- **Rationale citation:** Some systems (Wornow et al., Nievas et al.) instruct the LLM to cite specific sentences from the input to support decisions, improving both accuracy and human trust.

---

## 5. MODEL SELECTION

### 5.1 LLM Choices in the Literature

| Model | Type | Key Findings |
|---|---|---|
| **GPT-4 / GPT-4o** | Closed-source, proprietary | Consistently highest performance in head-to-head comparisons; best zero-shot capabilities; most expensive ($0.15–$15.88/patient depending on architecture) |
| **GPT-3.5 Turbo** | Closed-source, proprietary | Decent baseline; much cheaper ($0.02–$0.03/patient); good for retrieval augmentation and re-ranking |
| **LLaMA / LLaMA-2 / LLaMA-3** | Open-source (Meta) | Viable when fine-tuned; can be deployed behind hospital firewalls for PHI protection |
| **Qwen (1.5-14B, 2.5-7B)** | Open-source | OncoLLM (fine-tuned Qwen-1.5-14B) achieved 63% criterion accuracy at $0.17/pair vs GPT-4's 68% at $6.18/pair. Qwen2.5-7B used by Gueguen et al. for LLM re-ranking |
| **Mistral-7B / Hermes-2-Pro-Mistral-7B** | Open-source | Alkhoury et al. showed Hermes-2-Pro outperformed GPT-4 for structured biomarker extraction (F2=0.94 vs 0.29); excels at JSON output generation |
| **Domain-specific:** MedLlama, Meditron, Panacea | Open-source, medical | Panacea (fine-tuned Mistral-7B) pre-trained on 793K trial docs + 1.1M papers; MedLlama and Meditron explored but generally underperform GPT-4 without fine-tuning |

### 5.2 Prompting Strategies

| Strategy | Description | When to Use |
|---|---|---|
| **Zero-shot** | Task description only, no examples | Default for GPT-4 (strong baseline); sufficient for well-defined tasks |
| **Few-shot (1–2 examples)** | Include input-output examples in prompt | Helpful for GPT-3.5; diminishing returns with >2 examples; risk of overfitting to examples |
| **Chain-of-thought (CoT)** | Instruct model to reason step-by-step before answering | Used by TrialGPT for criterion-level matching; generates explanation → evidence → classification |
| **Prompt chaining** | Split task into subtasks with sequential prompts | Mixed results — improved GPT-4 for biomarker extraction but hurt GPT-3.5 (Alkhoury et al.) |

### 5.3 Fine-Tuning Approaches

| Approach | Description | Key Results |
|---|---|---|
| **Supervised Fine-Tuning (SFT)** | Train on labeled patient-criterion pairs | OncoLLM (Gupta et al.) trained for $2,688; Trial-LLAMA (Nievas et al.) distilled from GPT-4 |
| **Direct Preference Optimization (DPO)** | Optimize model using preferred vs rejected outputs | Alkhoury et al.: fine-tuning with DPO + synthetic data augmentation achieved F2=0.94 for biomarker structuring |
| **QLoRA** | Quantized low-rank adaptation for memory-efficient fine-tuning | Used by Alkhoury et al. with rank=2, scaling=4 on Hermes-2-Pro-Mistral-7B |
| **Instruction tuning on trial data** | Pre-train on large trial corpus before task-specific tuning | Panacea (Lin et al.) used 793K trial docs + 1.1M papers |

**Key insight:** GPT-4 generated synthetic data can effectively augment small manually annotated datasets for fine-tuning smaller open-source models. Alkhoury et al. showed that augmenting 92 manual samples with 80 GPT-4-generated synthetic samples dramatically improved fine-tuned model performance.

---

## 6. EVALUATION

### 6.1 Metrics by Task

| Task | Metrics | Notes |
|---|---|---|
| **Criterion-level matching** | Accuracy, Precision, Recall, F1, Sensitivity, Specificity | TrialGPT: 87.3% accuracy; Expert range: 87.6–91.6% |
| **Trial ranking** | NDCG@10, Precision@10, Reciprocal Rank, MAP@10 | TrialGPT: NDCG@10=0.7275 (best); Trial-LLAMA: 0.6636 |
| **Trial exclusion** | AUROC, AURPC | TrialGPT: AUROC=0.7979 |
| **Biomarker extraction** | Precision, Recall, F2 (weighted toward recall) | Hermes-FT-synth: F2=0.94 (inclusion), 0.94 (exclusion) in DNF |
| **Retrieval** | Recall@k, NDCG@1000 | TrialGPT-Retrieval: 90% recall at top 5.5% of collection |
| **Explanation quality** | % Correct / Partially Correct / Incorrect (human eval) | TrialGPT: 87.8% correct explanations |
| **Time efficiency** | Screening time reduction, cost per patient | TrialGPT: 42.6% screening time reduction |

### 6.2 Benchmark Datasets

| Dataset | Patients | Trials | Labels | Primary Use |
|---|---|---|---|---|
| SIGIR 2016 | 58 | 3,621 | Irrelevant / Potential / Eligible | Criterion-level + ranking |
| TREC 2021 CT | 75 | 26,149 | Irrelevant / Excluded / Eligible | Retrieval + ranking |
| TREC 2022 CT | 50 | 26,581 | Irrelevant / Excluded / Eligible | Retrieval + ranking |
| TREC 2023 CT | 50 | ~451K docs | Structured Q&A format | Retrieval + ranking |
| n2c2 2018 | 288 | 1 synthetic trial (13 criteria) | Met / Not Met | Criterion-level only |

### 6.3 Prospective vs. Retrospective Reality Gap

Gueguen et al. is the **only prospective study on real-world sequential patients** and found substantially lower performance than retrospective synthetic benchmarks:
- Mean precision across 4 tools: 0.33 (vs. retrospective meta-analysis reporting 90.5% sensitivity)
- 38% of patients had **no trials suggested** by any tool
- Most frequent error: **gene variant mismatches** (28–51% of false positives)
- LLM re-ranking improved NDCG@3 by ~5% (0.61 → 0.64)

---

## 7. COST & EFFICIENCY

| System | Cost per Patient-Trial Pair | Screening Time | Notes |
|---|---|---|---|
| **GPT-4 (no RAG)** | $6.18–$15.88 | 7.9–12.4 min | Highest accuracy but most expensive |
| **GPT-4 with RAG** | $0.02 | — | Unlu et al.: RAG dropped cost by ~800x |
| **GPT-3.5** | $0.02–$0.03 | 1.4–3 min | Good for retrieval/re-ranking steps |
| **OncoLLM (Qwen-1.5-14B)** | $0.17 | — | Training cost: $2,688 |
| **Llama-3.1-8B** | ~$929 for 10K patients | — | Open-source, self-hosted |
| **TrialGPT (full pipeline)** | — | 42.6% time reduction vs manual | Combined with human-in-the-loop |

---

## 8. KNOWN LIMITATIONS & OPEN CHALLENGES

1. **Data constraints:** Most studies use synthetic patients; real EHR data is scarce, hard to share, and messy. There is a significant performance gap between synthetic benchmarks and prospective real-world evaluation.

2. **Genomic biomarker matching:** The most frequent source of errors in prospective evaluation. Gene variant specificity (e.g., KRAS G12C vs G12V) requires precise extraction and matching that most tools currently fail at.

3. **Complex eligibility criteria:** Temporal constraints ("completed chemotherapy ≥4 weeks ago"), numerical thresholds ("LVEF ≥50%"), and multi-condition logic remain challenging for LLMs.

4. **Hallucinations:** LLMs can fabricate patient details or misinterpret negated statements. Ensemble approaches (Lai et al.: 5 independent predictions aggregated) and evidence citation help mitigate this.

5. **Context window limitations:** Full EHR histories can exceed LLM context windows. Retrieval and chunking strategies are essential but can lose chronological context.

6. **Trial status accuracy:** Recruitment status changes frequently and ClinicalTrials.gov updates lag behind reality.

7. **Interpretability:** Despite explanation generation, LLMs remain largely black-box. Chain-of-thought prompting and criterion-level decomposition improve but don't fully solve this.

8. **Equity & bias:** Underrepresentation of minorities, elderly, and rural populations in training data may perpetuate enrollment disparities. Model performance across diverse patient groups is rarely assessed.

9. **Standardization:** Heterogeneous evaluation metrics and datasets prevent meaningful cross-study comparison. The field needs consensus on benchmark tasks and metrics.

10. **Closed-source dependency:** 19/24 studies in Chen et al.'s review used OpenAI GPT models, raising concerns about reproducibility, data privacy, and vendor lock-in.

---

## 9. RECOMMENDED PIPELINE ARCHITECTURE

Based on the synthesis of all five papers, a practical pipeline would consist of:

```
Patient Data (EHR/notes/genomics)     Trial Data (ClinicalTrials.gov)
            │                                       │
            ▼                                       ▼
   ┌─────────────────┐                    ┌──────────────────┐
   │ Patient Data     │                    │ Criteria         │
   │ Processing       │                    │ Processing       │
   │ • NER extraction │                    │ • Parsing        │
   │ • Normalization  │                    │ • Normalization  │
   │ • Chunking       │                    │ • Rephrasing/DNF │
   │ • Summary gen    │                    │ • Biomarker      │
   └────────┬────────┘                    │   extraction     │
            │                             └────────┬─────────┘
            ▼                                       │
   ┌─────────────────┐                              │
   │ Retrieval        │◄─────────────────────────────┘
   │ • Keyword gen    │
   │ • BM25 + Dense   │
   │ • Hybrid fusion  │
   │ → Top ~500 trials│
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │ Criterion-Level  │
   │ Matching          │
   │ • Per-criterion   │
   │   LLM evaluation │
   │ • Explanation gen │
   │ • Evidence citing │
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │ Trial-Level      │
   │ Ranking           │
   │ • Linear aggr.   │
   │ • LLM aggr.      │
   │ • Feature combo   │
   │ • Exclusion logic │
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │ Human-in-Loop    │
   │ Review            │
   │ • Ranked trials  │
   │ • Explanations   │
   │ • Evidence sents │
   └─────────────────┘
```

---

## References

1. Jin Q, Wang Z, Floudas CS, et al. Matching patients to clinical trials with large language models. *Nature Communications* 15:9074 (2024).
2. Chen H, Li X, He X, et al. Enhancing patient-trial matching with large language models: A scoping review. *JCO Clinical Cancer Informatics* 9:e2500071 (2025).
3. Morrison BA, Sushil M, Young JS. A systematic review of trial-matching pipelines using large language models. Preprint (2025).
4. Gueguen L, Olgiati L, Brutti-Mairesse C, et al. A prospective pragmatic evaluation of automatic trial matching tools in a molecular tumor board. *npj Precision Oncology* 9:28 (2025).
5. Alkhoury N, Shaik M, Wurmus R, Akalin A. Enhancing biomarker based oncology trial matching using large language models. *npj Digital Medicine* 8:250 (2025).
