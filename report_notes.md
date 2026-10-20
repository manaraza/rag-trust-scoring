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

**Collateral false alarms (clean passages inside attacked n=1 groups)** (preview)
- 0.067 (NQ+HotpotQA) and 0.22 (MS MARCO) at the 5% threshold. Expect DistilBERT to remove more clean passages on MS MARCO in the end-to-end runs.

**Training**
- Six models (augmented and not augmented, seeds 42/43/44); training loss fell to about 0.01-0.02 (augmented) and 0.001-0.003 (not augmented) by epoch 3.

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
