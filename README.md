# RAG Trust-Scoring

COMPSCI 764 research project (Trustworthy Deep Learning, Project Idea 5.1): detecting adversarially retrieved passages in Retrieval-Augmented Generation before they reach the generator, with a lightweight per-passage trust score.

**Authors:** Samana and Aleena

## Main findings

- A DistilBERT detector on (question, passage) pairs, trained with shortcut-breaking augmentation, keeps AUROC at about 0.99 on NQ + HotpotQA and on the held-out MS MARCO corpus, including when the question prefix and the length and full-stop cues are removed.
- Without augmentation the same model learns the question-prefix shortcut (AUROC 0.73 / 0.41 on the shortcut-controlled pairing).
- A three-feature logistic regression (retrieval score, z-score, centroid distance) matches DistilBERT on unmodified in-distribution passages but fails once the prefix is removed and on MS MARCO (paired AUROC difference +0.07 and +0.47, CI excludes zero for all three seeds).
- Inside RAG, filtering with the augmented detector cuts single-poison attack success from 0.70 to 0.025 (NQ + HotpotQA) and from 0.77 to 0.14 (MS MARCO, the no-attack floor is 0.13). A simplified TrustRAG Stage-1 clustering defence removes no poison at n = 1.
- No detector generalises to a different attack (prompt injection, AUROC 0.33 to 0.44), and the validation threshold does not transfer to MS MARCO (22% of clean passages flagged).

Full numbers: `results/step6_eval/` (Table 1, Table 2, Fig 3, Fig 4, paired tests, sensitivity check, latency).

## Setup

| Component | Choice |
|---|---|
| Attack | PoisonedRAG black-box released passages (pinned commit `f660d72`): question + "." + attacker text |
| Retriever | `facebook/contriever`, mean pooling, dot product, exact top-5 |
| Generator | `Qwen/Qwen2.5-3B-Instruct`, fp16, greedy decoding |
| Datasets | NQ and HotpotQA (60/20/20 split by question, seed 42); MS MARCO (test only, never used for training or thresholds) |
| Detectors | DistilBERT (augmented headline, not augmented ablation; seeds 42/43/44); feature scorer (logistic regression); baselines: retrieval score, length, starts-with-query, ends-with-full-stop, simplified TrustRAG Stage-1 |
| Thresholds | 5% and 1% FPR on validation n = 0 clean passages, frozen before any test scoring |

## Pipeline (run in this order; Google Colab)

| Notebook | Runtime | What it does | Main outputs |
|---|---|---|---|
| `01_baseline.ipynb` | T4 | Undefended RAG baseline and attack check on NQ, HotpotQA, MS MARCO | `results/step1*/` |
| `02_build_dataset.ipynb` | CPU | Builds every evaluation group (n = 0 / 1 / 5) and shortcut-control variant; `build_core.py` | `data/passages.jsonl.gz`, `data/build_report.json` |
| `02b_features.ipynb`, `baselines_scorer_thresholds_.ipynb`, `step2_after_rebuild.ipynb` | GPU / CPU | Contriever features, feature scorer, baseline thresholds (frozen) | `data/frozen/` |
| `trustrag1-2.ipynb` | CPU | Simplified TrustRAG Stage-1 flags | flags (Drive) |
| `injection.ipynb` | CPU | Unseen-attack set (20 prompt-injection passages) | `data/injection.jsonl` |
| `03_distilbert.ipynb` | T4 | Trains the 6 DistilBERT models, sets thresholds, scores every row | `results/step3_distilbert/` |
| `04_rewrite_clean.ipynb` | T4 | LLM rewrites of clean test passages (style test) | `data/rewritten_clean*.jsonl`, `data/rewrite_*` |
| `03b_check_eval_core_and_export_features.ipynb` | CPU | Independent check of `eval_core.py`; exports feature scores | `results/step3_features/scores_features.jsonl` |
| `03c_latency.ipynb` | T4 | Latency per passage for every detector | `results/step3_features/latency.json` |
| `05_end_to_end_in_dist.ipynb`, `05_end_to_end_msmarco.ipynb` | T4 | Defences inside RAG (n = 0 / 1 / 5) | `results/step5_e2e/` |
| `06_final_evaluation.ipynb` | CPU | All detectors and baselines through one harness (`eval_core.py`): tables, figures, paired tests, sensitivity check, failure sheets | `results/step6_eval/` |

## Key documents

| File | Contents |
|---|---|
| `PLAN.md` | Project plan and schedule |
| `eval_protocol.md` | Evaluation protocol, frozen before any detector was run (amendments marked with dates) |
| `decisions.md` | Every design decision, deviation and check, with the reason for each |
| `report_notes.md` | Findings and agreed wording for the report |

## Reproducibility

- Every random step uses seed 42 (DistilBERT also 43 and 44); the dataset build is deterministic and checked by checksum (`decisions.md`).
- Thresholds are frozen on validation and asserted unchanged on every re-run.
- Large files (the uncompressed `passages.jsonl`, the six DistilBERT models, embedding caches) live on Google Drive under `rag_trust/`; everything needed to recompute the tables and figures from saved scores is in this repository.

## Running a notebook

Open [colab.research.google.com](https://colab.research.google.com), choose the GitHub tab, tick "Include private repos" and pick the notebook. Set Runtime, Change runtime type, to T4 GPU where the table above says T4. Each notebook mounts Google Drive and reads and writes `MyDrive/rag_trust/`.
