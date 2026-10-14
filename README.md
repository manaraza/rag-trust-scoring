# RAG Trust-Scoring

COMPSCI 764 research project (Trustworthy Deep Learning, Project Idea 5.1): detecting adversarially retrieved passages in Retrieval-Augmented Generation before they reach the generator.

**Authors:** Samana and Aleena

## Structure

| Path | Contents |
|---|---|
| `notebooks/` | Experiment notebooks, run in Google Colab (T4 GPU) |
| `results/` | Small outputs from each step: summaries, tables, figures |
| `decisions.md` | Log of design decisions and the reason for each |

Large data (corpus passages, per-question logs) is kept on Google Drive, not in this repository.

## Steps

1. `01_baseline.ipynb` — undefended RAG baseline and attack check (PoisonedRAG on Natural Questions).
2. Labelled dataset of clean and poisoned passages *(to do)*
3. Feature-based detector *(to do)*
4. DistilBERT detector *(to do)*
5. Evaluation and failure analysis *(to do)*

## Running a notebook

Open [colab.research.google.com](https://colab.research.google.com) → GitHub tab → tick "Include private repos" → choose the notebook. Set Runtime → Change runtime type → T4 GPU.
