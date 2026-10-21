# Report notes: agreed wording and findings

Running file of wording agreed during the experiments, to paste into the IEEE report.
Numbers marked **(preview)** come from the 03 preview (seed 42 CIs, before the rewritten-clean rows were added). The final numbers come from `06_final_evaluation.ipynb` (Table 1, Fig 3, Fig 4); update them there before submitting.

---

## 1. Detection results, DistilBERT (Results section)

**Augmentation removes the prefix shortcut (headline finding)** (preview)
- Without augmentation, DistilBERT learns the question-prefix cue: when the prefix is stripped from poisoned passages, TPR at the 5% FPR threshold drops to 0.31 (NQ+HotpotQA) and 0.29 (MS MARCO); when the question is prepended to clean passages, it flags 0.885 and 0.961 of them.
- With augmentation, both tests stay near the original: AUROC about 0.99, TPR 0.99 at 5% FPR.

**Shortcut-controlled pairing: good ranking, poor operating point** (preview)
- Augmented DistilBERT still ranks almost perfectly (AUROC 0.993 NQ+HotpotQA, 0.990 MS MARCO), but at the frozen threshold it flags 0.27 and 0.39 of truncated clean passages.
- Suggested sentence: "The detector separates truncated clean passages from prefix-free poison, but the threshold fixed on unmodified validation passages does not transfer to them."

**The threshold does not transfer to MS MARCO** (preview)
- NQ+HotpotQA test FPR at the 5% threshold: 0.065 (13 of 200), within sampling noise of the 5% target.
- MS MARCO FPR: 0.22, although AUROC stays at 0.994. Thresholds are frozen by protocol and not recalibrated, so this is reported as a finding: the ranking transfers across corpora, the operating point does not.

**The unseen injection attack defeats the detector** (preview)
- AUROC 0.32, TPR 0.00 at 5% FPR (NQ+HotpotQA). Below 0.5 means injected passages are rated cleaner than real clean passages.
- Suggested sentence: "DistilBERT learns what PoisonedRAG passages look like rather than what makes a passage adversarial."

**LLM-style clean passages: augmentation moved the model onto a style shortcut** (preview, after the rewrites were added)
- Augmented DistilBERT flags 0.27 (NQ+HotpotQA) and 0.48 (MS MARCO) of rewritten clean passages at the 5% threshold, against 0.065 and 0.22 for the same passages in their original wording; AUROC stays high (0.990, 0.975).
- Without augmentation the effect is much smaller: FPR 0.05 (NQ+HotpotQA), 0.22 (MS MARCO).
- Suggested sentence: "Removing the prefix cue did not make the detector cue-free: deprived of the prefix, the augmented model leans more on LLM writing style, so clean passages rewritten by an LLM are flagged about four times as often as their originals."
- Caveat for the same paragraph: 8 of 20 spot-checked rewrites also changed a fact (section 3), so the llm_style negatives differ in content as well as style, most often on MS MARCO.
- 4 of the 580 passing "rewrites" are identical to their original (3 MS MARCO, 1 NQ: a 3-word passage, an address listing, an airport-code list and a forum snippet the model left unchanged); they were kept, which can only lower the llm_style FPR slightly (at most 1 of 186 and 3 of 394 negatives).

**Collateral false alarms (clean passages inside attacked n=1 groups)** (preview)
- 0.067 (NQ+HotpotQA) and 0.22 (MS MARCO) at the 5% threshold. Expect DistilBERT to remove more clean passages on MS MARCO in the end-to-end runs.

**Training**
- Six models (augmented and not augmented, seeds 42/43/44); training loss fell to about 0.01-0.02 (augmented) and 0.001-0.003 (not augmented) by epoch 3.

---

## 1a. FINAL detection results (notebook 06, 21 Oct): these replace the preview numbers in section 1

**Paired AUROC difference, DistilBERT (augmented) minus feature scorer (pre-registered; same rows, bootstrap by question, 3 seeds):**

| | Original | Shortcut-controlled |
|---|---|---|
| NQ + HotpotQA | +0.004 to +0.005, CI includes 0 (all 3 seeds) | **+0.067 to +0.073**, CI excludes 0 (all 3 seeds) |
| MS MARCO (held out) | **+0.099 to +0.101**, CI excludes 0 | **+0.463 to +0.467**, CI excludes 0 |

- Suggested sentence: "On unmodified in-distribution passages the three-feature scorer is as good as DistilBERT (difference +0.005, CI includes zero), but DistilBERT is significantly better once the prefix and length cues are removed (+0.07) and on the held-out corpus (+0.10, and +0.47 with the cues removed), for all three seeds."

**Augmentation effect (additional, not pre-registered; augmented minus not augmented, same seed):**
- Original: -0.002 to -0.006 (augmentation costs almost nothing; significant in 4 of 6 cases because the CIs are very tight).
- Shortcut-controlled: **+0.24 to +0.31 (NQ+HotpotQA), +0.56 to +0.59 (MS MARCO)**, CI excludes 0 for every seed.

**Why the feature scorer and the retrieval score fail under shortcut control:** both rely mostly on the retrieval score, and the question prefix is what makes poison score high. With the prefix removed:
- Feature scorer AUROC: 0.993 to 0.769 (NQ+HotpotQA) and 0.893 to 0.351 (MS MARCO).
- Retrieval score AUROC: 0.994 to 0.907 (NQ+HotpotQA shortcut-controlled) and 0.847 to 0.505 (MS MARCO).
- Augmented DistilBERT stays at 0.992-0.998.

**Feature scorer at n=5 (outside training distribution):** AUROC 0.900 (NQ+HotpotQA) and 0.603 (MS MARCO), against 0.993 and 0.893 at n=1.

**Unseen injection attack (NQ+HotpotQA):** DistilBERT augmented 0.33, not augmented 0.32, feature scorer 0.44 [0.30, 0.59]. No detector generalises to it.

**Baselines (Table 1):** starts-with-query is perfect on the original pairing (AUROC 1.0) and at chance (0.5) once the prefix is removed, which is the prefix shortcut in one line; ends-with-full-stop 0.68 / 0.55 original, 0.50 shortcut-controlled; length 0.80 / 0.88 original but catches nothing at its threshold; TrustRAG flags 0% of poison passages at n=1.

**Sensitivity check (protocol §10):** excluding the 5 hand-check-flagged test questions changes n=1 ASR by at most 0.016 for any defence (e.g. none 0.700 to 0.684 NQ+HotpotQA, 0.770 to 0.763 MS MARCO); no conclusion changes.

**Failure sheets:** augmented DistilBERT has 5 misses and 291 false alarms across the original and shortcut-controlled test pairings (sheet: all 5 misses + 20 false alarms); the feature scorer has 567 misses (mostly prefix-removed poison) and 158 false alarms (sheet: 20 + 20).

---

## 1b. End-to-end results (Results section; NQ+HotpotQA shard, 40 test questions, 5% FPR threshold)

- Undefended numbers reproduce Step 1 exactly: ASR 0.05 / 0.70 / 0.975 at n = 0 / 1 / 5. Reuse check: 10/10 regenerated answers identical to Step 1, so answers for groups where a defence removed nothing are reused.
- **DistilBERT (augmented, seed 42) cuts attack success at n=1 from 0.70 [0.55, 0.83] to 0.025 [0.00, 0.08]** and at n=5 from 0.975 to 0.05.
- At n=5 it removes all five passages in 0.875 of questions, so the generator answers with no context; "says correct" is 0.375 there (the model falls back on its own knowledge).
- Cost on clean questions (n=0): on average 0.325 of 5 clean passages removed per question (about 6.5% of passages, in line with the 6.5% test FPR), and clean-answer F1 is unchanged (retention 1.005).
- Suggested sentence: "On NQ and HotpotQA, filtering with the augmented detector reduces single-poison attack success from 70% to 2.5% without lowering clean-answer quality."
- **All four defences, NQ+HotpotQA (ASR substring, 95% CI):**

| Defence | n=0 | n=1 | n=5 | Clean removed per question, n=0 / n=1 (mean count) | F1 retention (n=0) |
|---|---|---|---|---|---|
| None | 0.05 | 0.70 [0.55, 0.83] | 0.975 | 0 / 0 | 1 |
| DistilBERT (aug, s42) | 0.05 | 0.025 [0.00, 0.08] | 0.05 | 0.33 / 0.28 | 1.005 |
| Feature scorer | 0.075 | 0.025 [0.00, 0.08] | 0.525 [0.40, 0.65] | 0.25 / 0.05 | 0.973 |
| TrustRAG Stage-1 (simplified) | 0.05 | 0.70 [0.55, 0.83] | 0.875 [0.78, 0.95] | 0 / 0.05 | 1.000 |

- **Both per-passage detectors stop the single-poison attack** (0.70 to 0.025). The feature scorer removes fewer clean passages inside attacked groups (0.05 vs 0.28 per question) but loses a little clean-answer F1 (0.973).
- **The feature scorer fails at n=5** (ASR 0.525; it keeps 38% of poison). This is outside its training distribution (it was trained on n=0 and n=1 groups only, protocol §5): with five poisoned passages its group features (z-score, distance from the other passages) no longer single out poison. Report it with that caveat.
- **TrustRAG does nothing at n=1** (keeps every poisoned passage; ASR unchanged at 0.70), which is the single-passage gap that motivates per-passage scoring. **Even at n=5, the setting it was designed for, it only removes about half the poison (keeps 45.5%), and ASR stays at 0.875**, because the remaining poisoned passages are enough to mislead the generator.
- Suggested sentence: "A clustering defence cannot see a single poisoned passage, and at five it removes only part of the cluster; per-passage scores close the n=1 gap, but the feature scorer, which relies on group statistics, breaks down when every retrieved passage is poisoned."
- The run notebook's "clean passages removed" column is a **mean count per question**; Table 2 in notebook 06 also gives it as a rate (removed / clean passages in the group), adds ASR given poison retrieved, and latency per passage.
- **All four defences, MS MARCO (held out; 100 test questions; ASR substring, 95% CI):**

| Defence | n=0 | n=1 | n=5 | Clean removed per question, n=0 / n=1 (mean count) | F1 retention (n=0) |
|---|---|---|---|---|---|
| None | 0.13 | 0.77 [0.68, 0.85] | 0.93 | 0 / 0 | 1 |
| DistilBERT (aug, s42) | 0.14 | 0.14 [0.08, 0.21] | 0.13 | 1.11 / 0.89 | 0.901 |
| Feature scorer | 0.14 | 0.22 [0.14, 0.31] | 0.91 [0.85, 0.96] | 1.04 / 0.50 | 0.851 |
| TrustRAG Stage-1 (simplified) | 0.13 | 0.79 [0.71, 0.87] | 0.87 [0.80, 0.93] | 0.25 / 0.32 | 0.981 |

- **The n=0 rate is a floor, not an attack.** With no poison at all, the attacker's answer already appears in 0.13 of MS MARCO answers by substring (0.08 whole word), mostly short answers such as "no" matching inside other words. **DistilBERT brings n=1 and n=5 attack success down to that floor** (0.14 and 0.13; it keeps 0% of poison).
- **The price on MS MARCO is clean passages:** DistilBERT removes on average 1.1 of 5 clean passages per question at n=0 (about 22%, matching its 22% MS MARCO false-alarm rate), and clean-answer F1 drops to 0.90 of undefended. The feature scorer removes about as many (1.0 per question) and loses more F1 (0.85), while still keeping 21% of poison at n=1. This is the threshold-transfer problem from Table 1 showing up inside RAG.
- **TrustRAG again has no effect at n=1** (0.79; keeps all poison) and little at n=5 (0.87), at a small clean cost (0.98 retention).
- **The feature scorer fails at n=5 on MS MARCO too** (0.91; keeps 64% of poison), outside its training distribution.
- Suggested sentence: "On the held-out MS MARCO corpus the augmented detector still removes every poisoned passage, bringing attack success down to the rate seen without any attack, but the validation threshold flags about a fifth of clean passages, which costs 10% of clean-answer F1."

---

## 1c. Latency per passage (Table 2 / Discussion; Tesla T4, group of 5, median of 200 test groups)

| Detector | Marginal (retriever embeddings already computed) | Standalone (incl. Contriever embedding) |
|---|---|---|
| DistilBERT (aug s42) | 1.86 ms | 1.86 ms (needs no embeddings) |
| Feature scorer | 0.11 ms | 9.9 ms |
| TrustRAG Stage-1 | 1.03 ms | 11.0 ms |

- All three are cheap next to generation (Qwen2.5-3B takes a few hundred ms per answer on the same GPU; exact median from Step 1 records is printed by notebook 06).
- Suggested sentence: "Every detector adds at most about 2 ms per passage once the retriever's embeddings are available; DistilBERT needs no embeddings at all, so its 1.9 ms is also its standalone cost, while the feature scorer and TrustRAG cost about 10-11 ms per passage if the embeddings must be computed."
- Timing only: the scaler's feature-name warning in notebook 03c does not affect any score.

---

## 2. Feature scorer and baselines (Aleena's findings)

- Retrieval score alone: train AUROC 0.985; val TPR 0.97 at 5% FPR.
- Feature scorer: val TPR 0.98 at 5% FPR; coefficient on f1 (retrieval score) 0.855, so it relies mostly on the retrieval score.
- Length baseline: flags 0 val poison passages at its thresholds.
- Simplified TrustRAG Stage-1: removes 0 poison passages in train n=1 groups and fires on 10% of val n=0 groups.

---

## 3. Rewritten-clean (llm_style) test (Method + Limitations)

**Deviation: corrected checks and one regeneration round** (also in decisions.md)
> Round 1 passed 390/700 (NQ 34/100). Reading the failures showed that the number check counted list markers ("2. Purchase...") and citation markers ("[34]") as facts and rejected numbers written as words, while it ignored numbers the model added. Version 2 ignores those markers, accepts written-out numbers, and also rejects added numbers; the length rule (0.7-1.3x) is unchanged. Under v2, round 1 passes 456/700. Failures were regenerated once with a fixed sentence-by-sentence prompt (greedy), under the same checks; passages failing both rounds are excluded. This was decided after seeing round-1 results, so it is reported as a deviation. Both raw files are committed.

**Result:** 580 of 700 rewrites pass (456 round 1, 124 round 2): NQ 95/100, HotpotQA 91/100, MS MARCO 394/500. The 120 excluded failed for real reasons (64 dropped a number, 65 wrong length, 19 added a number) and are never hand-fixed.

**Report sentence (Method):**
> "One deviation from the registered protocol: after the first round of rewrites, the automatic number check was corrected (it had treated list and citation markers as facts) and tightened (it now also rejects added numbers), and failed rewrites were regenerated once with a sentence-by-sentence prompt under the same checks. 580 of 700 rewrites passed."

**Spot-check (20 rewrites, seed 42): 9 ok, 8 fact_changed, 3 other**

| | ok | fact_changed | other |
|---|---|---|---|
| MS MARCO (12) | 4 | 7 | 1 |
| NQ + HotpotQA (8) | 5 | 1 | 2 |
| Round 1 (16) | 6 | 7 | 3 |
| Round 2 (4) | 3 | 1 | 0 |

**Report sentence (Limitations):**
> "A spot-check of 20 rewrites (seed 42) found 9 faithful, 8 with at least one changed fact and 3 with a shift in meaning; 7 of the 8 fact changes were MS MARCO passages, against 1 of 8 for NQ and HotpotQA. The automatic checks catch changed digits but not changed claims. The rewrites remain non-adversarial clean passages, so their label is unchanged; the llm_style pairing tests sensitivity to LLM writing style, not to factual accuracy, and MS MARCO llm_style results should be read with this in mind."

**Examples worth one line each (Discussion):**
- **The model answered the question inside the passage.** An MS MARCO passage asked how Martin Luther King's speech related to the Universal Declaration of Human Rights and never answered; the rewrite stated that the speech "connected his vision... to the principles of the UDHR". This is the closest the rewrites come to how poison behaves.
- **The checks passed a reversed claim.** In the Ireland passage every number survived (1922, 1801, 1169, 1497, 1867), but "British control can be traced back to 1169" became "Ireland can trace its independence back to 1169". This is why a human spot-check is needed.
- **Long passages drift most.** The longest passage (Big Blue Nation) lost a qualifier ("regular season" undefeated), dropped place names and added content ("sea of blue").
- **Round 2 did not lower quality** (1 of 4 fact changes against 7 of 16 in round 1).

**Spot-check rule used (Method, one sentence):** names, numbers, dates, places and claims count as facts; a missing, changed or added fact is `fact_changed`; all facts kept but meaning or emphasis shifted (e.g. a dropped hedge) is `other`; wording-only changes are `ok`.

---

## 4. Evaluation harness (Method + contribution breakdown)

- One shared harness (`eval_core.py`) scores every detector and baseline by the same rules: pairings from protocol section 7, TPR/FPR at frozen 5% and 1% thresholds (score above threshold = flagged), bootstrap 95% CIs (2,000 resamples, seed 42, resampling questions within each dataset), paired AUROC difference per DistilBERT seed against the feature scorer.
- Yes/no baselines (starts-with-query, ends-with-full-stop, TrustRAG) report TPR and FPR at their fixed rule; TrustRAG gets no AUROC.
- **Independent check** (decisions.md): eval_core.py independently checked on val by Aleena (AUROC, TPR, FPR match sklearn to 3 dp; pairings match protocol section 7 on a synthetic set). Her check found the n=5 prefix-removed groups were not evaluated; pairing `prefix_poison_side_n5` added on 20 Oct, before any test result for it was seen.
- **Contribution statement (draft):** "Evaluation harness: initial implementation drafted with the DistilBERT pipeline; reviewed against the protocol, independently validated and run by Aleena."

---

## 5. Reproducibility (Method, one or two sentences)

- Final dataset: 32,252 rows (20 injection passages, 580 rewrites), protocol checks all passed. md5 `passages.jsonl` a40ce17357660d162bd5c017c9663ef6, `passages.jsonl.gz` f06f566724e20caed66ca96d07da0023.
- Adding the rewrites changed no existing row: with the rewritten_clean groups removed, the file is byte-identical to the injection build (29,352 rows, md5 b97676483e9001416c432b392017f20d), and the build without injection reproduces md5 06d4d3bc7676471f57a6318db407ee0b.
- Suggested sentence: "The dataset build is deterministic (seed 42); each added input only appends groups, which we verified by checksum, so thresholds frozen earlier remain valid."

---

## 6. Limitations (collect here)

- NQ test set is small (20 questions); wide CIs.
- Single attack family (PoisonedRAG black-box); the unseen injection attack shows the detector does not generalise beyond it.
- One corpus per dataset; MS MARCO shows the threshold does not transfer across corpora.
- Injection passages differ in length from PoisonedRAG poison (length confound).
- Results depend on the fixed generator prompt.
- Rewritten-clean test: 120 of 700 rewrites excluded by the automatic checks; 8 of 20 spot-checked rewrites changed a fact (mostly MS MARCO).
- Hand-check-flagged test questions (NQ test442, test53; MS MARCO 156641, 416509, 577564): sensitivity check reported in 06.
