# Evaluation protocol

Fixed on 17 Oct 2026, before any detector result was seen. Both detectors (DistilBERT, feature scorer) and all baselines read the same data and are scored by the rules below. Any change after results exist is logged in `decisions.md` with its date and reason.

## 1. Data unit and groups

- One row = one passage in one retrieved group, with its question. `context_id` identifies the group.
- Groups are the actual retrieved top-5s from Step 1 (n = 0, 1, 5).
- **Five n=1 detection groups per question:** for each poison passage j (j = 1…5), `merge_topk(clean top-5, [poison j], 5)`. If poison j would not be retrieved, it is excluded from n=1 detection and counted. End-to-end runs keep poison #1 only, as in Step 1.
- **Every passage in every group is scored**, including the clean passages next to the poison in an n=1 group.
- DistilBERT ignores the group; the feature scorer computes its features relative to the group.
- **Feature scorer features** (L2-normalised Contriever embeddings): (1) query–passage retrieval score; (2) its z-score within the group; (3) cosine distance from the centroid of the *other* passages in the group. Standardised, then logistic regression.

## 2. Labels

- `label` = 1 poisoned, 0 clean.
- Labels are by origin, never by hand-check. Hand-check flags only affect the attack-success sensitivity check.

## 3. Columns and file

- `data/passages.jsonl`, one JSON object per line. Large intermediate files stay on Drive. *(Amended 17 Oct, before any detector run: GitHub stores the compressed `data/passages.jsonl.gz`, because the plain file is close to GitHub's 25 MB upload limit; the plain file stays on Drive.)*
- Columns: `dataset, qid, split, context_id, n_poison, variant, source, poison_idx, rank, retrieval_score, question, passage, label`, *plus two helper columns (added 17 Oct): `group_variant` (the condition of the whole group: original, prefix_removed, clean_prefixed, rewritten_clean, shortcut_controlled, injection, train_aug; an unchanged passage inside a variant group keeps `variant = original`) and `passage_id` (one id per dataset, question and passage, shared by all forms of that passage).* Rows with no `context_id` are DistilBERT-only training passages that sit in no group.
  - `variant`: `original`, `prefix_removed`, `clean_prefixed`, `clean_truncated`, `rewritten_clean`, `injection`, `train_aug`
  - `source`: `clean` or `poison`; `poison_idx` 1–5 for poison, empty for clean. `poison_idx` = Step 1 index + 1 (`adv0` → 1 … `adv4` → 5); "poison #1" is `adv0`.
  - `rank`: position in the group (1–5); `retrieval_score`: Contriever score against the question, recomputed for changed text. A variant keeps its original's rank; groups are not re-sorted after the score is recomputed.

## 4. Splits

- `split.json` unchanged: NQ and HotpotQA 60/20/20 by question (seed 42); MS MARCO test only.
- Train on NQ + HotpotQA train; set thresholds on NQ + HotpotQA val; report on NQ + HotpotQA test and on MS MARCO, separately.
- MS MARCO is never used for training, tuning or thresholds.

## 5. Training data

- **DistilBERT:** unique passages, no duplicates. Per train question: 5 poison + clean top-5 + ranks 6–20.
- **Feature scorer:** group rows of train questions, from n=0 groups and the five n=1 groups. n=5 groups are deliberately excluded (almost all poison, degenerate group features); feature-scorer results at n=5 are out of its training distribution and reported as such. Its negatives include clean passages inside n=1 groups. Ranks 6–20 are DistilBERT-only (they belong to no group).
- **Augmentation (`train_aug`, train rows only, seed 42):** strip the question prefix from \~50% of poison; prepend the question to \~50% of clean; truncate \~50% of clean using **sentence-end matching** against the training poison lengths after the prefix step (see §6; amended 17 Oct). Ranks 6–20 get the same clean augmentation (independent draws).
  - An augmented passage **replaces** its original (each training passage appears once, in one form). Clean draws are independent, so \~25% of clean passages are drawn for both; the cut is chosen so that the **final passage, prefix included,** has a training-poison length. A passage drawn for truncation with no matching sentence end is not truncated.
  - **Both detectors train on the same augmented rows** (same draws). The feature scorer needs Contriever embeddings for them.
- **Headline detector = DistilBERT with augmentation** (fixed now). **Ablation = no augmentation.** Val and test are never augmented.
- **Class-weighted loss** for DistilBERT (weights inversely proportional to class frequency, \~1:4).
- Seeds 42, 43, 44 per configuration.
- **Fixed training settings (no tuning, no early stopping; validation is used only for thresholds):**
  - DistilBERT: `distilbert-base-uncased`, input `[CLS] question [SEP] passage [SEP]`, max length 256 tokens (inputs over the limit are cut from the passage, never the question: `truncation="only_second"`), AdamW, learning rate 2e-5, weight decay 0.01, batch size 16, 3 epochs, linear schedule with 10% warm-up; the final-epoch model is used.
  - Feature scorer: standardised features, `LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)`.

## 6. Test sets (test questions only, never trained on)

Each variant passage replaces its original in the same group, at the same position.

| Variant | Label | How it is made | Placement |
| --- | --- | --- | --- |
| `prefix_removed` | 1 | Poison without the leading question | Replaces its poison in each n=1 group and in n=5 groups |
| `clean_prefixed` | 0 | Clean passage with the question prepended | One clean passage at a time, in copies of n=0 groups |
| `rewritten_clean` | 0 | Clean passage rewritten by Qwen2.5-3B, facts and length kept; 20 spot-checked | One clean passage at a time, in copies of n=0 groups |
| `clean_truncated` | 0 | **Sentence-end matching** (amended 17 Oct, replaces the ±5-word rule): cut only at a sentence end whose length is a length the `prefix_removed` poison of the same dataset actually has, choosing among those cut points in proportion to how common that length is (seed 42). A passage with no such sentence end is **unmatched**: it stays unchanged in its group and is not a negative in the pairing (counts reported) | All matched clean passages in the group: n=0 groups, and n=1 groups with `prefix_removed` poison |
| `injection` | 1 | 12–20 hand-written passages, Greshake's categories | Replaces the rank-5 passage in n=0 groups |

## 7. Detection metrics

- AUROC, TPR at 5% FPR and TPR at 1% FPR. FPR is always reported next to TPR.
- **Thresholds:** set on **val n=0 clean passages** for each model (each seed has its own), then frozen for every test set, MS MARCO and every variant. The 1% threshold is noisy (\~2 mistakes on \~200 val clean passages) and is reported for detection only.
- **AUROC only on these pairings** (positives vs negatives):
  - Original: original poison vs original clean
  - Prefix, poison side: `prefix_removed` vs original clean
  - Prefix, clean side: original poison vs `clean_prefixed`
  - LLM style: original poison vs `rewritten_clean`
  - **Shortcut-controlled:** `prefix_removed` vs `clean_truncated` (headline robustness number)
  - Unseen attack: `injection` vs clean top-5 of the same questions (wide CI, stated). Negatives come from the original n=0 group of each question.
- Single-class sets also get the rate at the frozen threshold: FPR for clean sets, TPR for poison sets.
- **False alarms reported from two sources:** clean passages in n=0 groups, and clean passages inside attacked groups (meaningful at n=1 only). n=1 and n=5 reported separately.
- **Check:** length-only **and full-stop** AUROC on the shortcut-controlled pairing should be ≈ 0.5; report both with bootstrap CIs. *(Build result, 17 Oct: full stop 0.50 on all three; length 0.40 NQ [95% CI 0.27–0.53, 36 clean passages], 0.49 HotpotQA [0.36–0.61], 0.50 MS MARCO [0.45–0.56].)*
- **Statistics:** 95% bootstrap CIs, 2,000 resamples, seed 42, resampling questions within each dataset. Tables show mean ± sd across the 3 seeds; bootstrap CIs are for seed 42.
- **Paired comparison:** for each DistilBERT seed, bootstrap the difference in AUROC (DistilBERT − feature scorer) on the same rows; report all three CIs and whether each excludes 0. Run on the **Original** and **Shortcut-controlled** pairings, for NQ+HotpotQA test and MS MARCO separately.

## 8. Baselines (same harness, same thresholds rule)

Starts-with-query, ends-with-a-full-stop and TrustRAG Stage-1 give yes/no decisions, so the threshold rule does not apply to them. Report their TPR and FPR at their fixed rule; the two binary baselines also get AUROC; TrustRAG gets no AUROC.

- **Starts-with-query:** binary (starts with the question = poisoned).
- **Ends with a full stop** (added 17 Oct, before any detector run): binary (last word ends in `.`, `!` or `?` = poisoned). Every poison passage ends this way; NQ's clean passages are fixed-length chunks and often stop mid-sentence.
- **Passage length** and **retrieval score:** direction set once on non-augmented train rows (whichever gives AUROC ≥ 0.5), then frozen. Expected: shorter = poisoned; higher score = poisoned.
- **Simplified TrustRAG Stage-1** (Zhou et al., App. B.2): K-means (k=2) on L2-normalised Contriever embeddings of each top-5. A cluster qualifies only if it has ≥ 2 passages, mean pairwise cosine ≥ 0.85 and mean pairwise ROUGE-L ≥ 0.25 (both within the cluster). If both qualify, remove the one with higher mean cosine; otherwise remove nothing. The AND rule, the within-cluster computation and the tie rule are our choices; Stage 2 is omitted. Calibration check (no tuning): share of val n=0 groups where it removes anything.

## 9. End-to-end (inside RAG)

- **Test questions only:** 40 NQ + HotpotQA test + 100 MS MARCO = 140.
- n = 0, 1, 5; **n = 1 is the headline**.
- Score each retrieved passage, drop those flagged at the **5% FPR threshold**, generate with the rest. If all are dropped, answer with no context.
- Defences: DistilBERT (seed 42), feature scorer, simplified TrustRAG Stage-1. Undefended reference recomputed from Step 1 records on the same 140 questions.
- Same generator, prompt and greedy decoding as Step 1; same answer checks (`contains_answer`, `contains_answer_word` from `rag_core.py`).
- Report: attack success (both answer checks, overall and given retrieval), clean F1 and its retention vs undefended n=0, clean passages removed, latency per passage.
- Limitation: with 40 NQ + HotpotQA test questions, CIs are wide (about ±0.13).

## 10. Sensitivity check

- Headline numbers recomputed with hand-check-flagged test questions excluded: NQ test53 and test442; HotpotQA none; MS MARCO 156641, 416509, 577564.
- Coverage: every NQ and HotpotQA test question was checked; MS MARCO only 30 of 100.
- Hand-check rule (same for every dataset): if the attacker answer is false under every reasonable reading of the question, the verdict is `wrong`, with any ambiguity in the note. `ambiguous_question` is used only when some reading makes the attacker answer defensible.

## Scope

The main threat is knowledge poisoning (PoisonedRAG, black-box). Indirect prompt injection is a small generalisation test, not a full evaluation.