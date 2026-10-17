# Project plan v2.2 — Sat Oct 17 to Sun Oct 25

**Deadline:** IEEE report (10–12 pp + refs) **and** recorded presentation (8–10 min, 4–5 min each) — **Sun Oct 25. Aim to submit by Sun midday.**
Replaces `PLAN.md` v1. v2.1 added the review fixes (pairings, length-matched truncation, length augmentation, TrustRAG's real removal rule, seeds, look-ahead log). v2.2: every passage scored in every group; headline fixed now (val sets thresholds only); prefix-removed lengths in the table; per-seed paired tests; corrected hand-check counts; end-to-end on test questions only (≈1,260 generations); group placement of variant passages; training vs evaluation counts; feature-scorer training rows; class-weighted loss; five n=1 detection groups per question; training length targets from actual training poison; both detectors on the same train_aug rows; thresholds on val n=0 clean; seeds 42/43/44; collateral FPR at n=1 only; one-at-a-time vs whole-group replacement; baseline direction frozen on train; ranks 6–20 augmented; feature scorer trains on n=0 + n=1 groups only; TrustRAG application details stated as ours; NQ hand-check coverage corrected.

---

## 1. Where we are

| Done | Evidence |
|---|---|
| Undefended baseline on NQ, HotpotQA, MS MARCO | Gate passed on all three. ASR at n=1: NQ 0.67, HotpotQA 0.75, MS MARCO 0.77, against n=0 controls of 0.03–0.13 |
| Retrieval of a single poisoned passage | 98–100% at n=1 |
| Hand-checks (30 questions per dataset) | **HotpotQA:** all 20 test + 10 others, 30/30 wrong. **MS MARCO (final):** 30 test questions, 27 wrong, 3 ambiguous (156641 hertz, 416509 asphalt, 577564 Camp Pendleton). **NQ (final, re-judged by Aleena):** all 20 test + 5 train + 5 val; 27 wrong, 1 also valid (test347, train), 2 ambiguous (test442, test53, both test) |
| Whole-word answer check | Done: in `rag_core.py`, notebook cell identical; committed with the protocol freeze |
| Evaluation protocol | **Agreed and frozen, Sat 17 Oct** (`eval_protocol.md`). Later changes go in `decisions.md` with date and reason |

---

## 2. What earns the marks

| Marks | What the marker looks for | How we cover it |
|---|---|---|
| Method 8% | Clear method, justified decisions, sound design, appropriate statistics | Protocol fixed before results; split by question; threshold frozen on val; bootstrap CIs; paired difference test; shortcut controls with defined pairings |
| Results 5% | More than accuracy; comparison with existing approaches; where and why it fails | 6 evaluations, 4 baselines incl. TrustRAG-style, end-to-end ASR **and** clean retention, failure analysis |
| Intro / RW / Abstract / Conclusion 8% | RQ + hypothesis + contributions; RW regrouped; limitations + future work | Writing schedule in §6 |
| Style 2% | Clear, within page limit, good figures | Two read-through passes |
| Presentation 10% | Problem, why it matters, idea, results, conclusions | Recorded Sat Oct 24 |

**Principle:** a small, controlled experiment we can explain beats a large one we can't. Every claim is written as "we test whether…" until the numbers exist.

---

## 3. Settle today, then commit `eval_protocol.md`

**From the protocol review**
1. **Context definition: score every passage in every group.** Groups are the actual retrieved top-5s from the Step 1 runs (n=0, 1, 5), identified by a `context_id` column. Every passage in every group gets a score, **including the 4 clean passages that sit next to the poison in an n=1 group**. Otherwise clean passages would only ever be scored in clean groups and poison only in attacked groups, and the feature scorer could separate them by the *group*, not the passage.
   - **Hits** (TPR) come from poison in attacked groups.
   - **False alarms** (FPR) come from two sources, reported separately: clean passages in **n=0 groups** (what a normal user loses) and clean passages **inside attacked groups** (collateral damage).
   - Report n=1 and n=5 separately. DistilBERT ignores the group, so its score for a passage is the same in every group.
   - **Collateral FPR is meaningful only at n=1.** With 98–100% retrieval, an n=5 group is almost all poison, so it has nearly no clean passages to score. At n=5 the feature scorer's group features also see an almost all-poison group; say so when reporting n=5.
   - **Five n=1 detection groups per question.** Step 1's n=1 group holds only poison #1, which would leave 40 in-distribution and 100 MS MARCO positives. For detection scoring, build one n=1 group per poison passage j (j = 1…5): `merge_topk(clean top-5, [poison j], 5)`, so poison j sits at its real retrieved rank with the clean passages it would actually be retrieved with. If poison j would not be retrieved, it is excluded from n=1 detection and counted. This gives up to 200 in-distribution and 500 MS MARCO positives, with no generation. Bootstrap by question accounts for the shared clean passages. **End-to-end runs keep poison #1 only**, as in Step 1.
2. **One filter threshold for end-to-end runs: 5% FPR on val.** 1% is reported for detection only, flagged as noisy (~2 mistakes on ~200 val clean passages).
3. **Add `rewritten_clean`** (test only, label 0): Qwen rewrites clean test passages, facts unchanged, length kept; spot-check 20. Tests the *LLM-style* shortcut.
4. **Drop** sceptical prompting, the prompted judge and the fusion ablation.

**Length shortcut (checked on our own data)**

| Dataset | Clean top-5, median words (p10–p90) | Poison with prefix | Poison, prefix removed | Length-only AUROC: with prefix / prefix removed |
|---|---|---|---|---|
| HotpotQA | 78 (13–149) | 42 (35–53) | **28 (25–30)** | 0.71 / **0.79** |
| MS MARCO (held out) | 51 (31–93) | 32 (29–36) | **27 (25–30)** | 0.88 / **0.92** |

Removing the prefix makes length an even stronger giveaway, so `prefix_removed` alone does not neutralise shortcuts.

5. **`clean_truncated`** (test only, label 0). It is paired with `prefix_removed` poison, so it is matched to **the prefix-removed lengths** (~25–30 words), not the 42-word passages with prefix. **Amended 17 Oct (before any detector run): sentence-end matching.** Cut only at a sentence end whose length is a length the `prefix_removed` poison of the same dataset actually has, choosing among those cut points in proportion to how common that length is (seed 42). Passages with no such sentence end are *unmatched*: unchanged in their group, not negatives in the pairing. Why: the ±5 rule gave length AUROC 0.39–0.49 and left 42–67% of truncated clean passages ending mid-sentence, while every poison passage ends with a full stop.
   *Check (build result):* full-stop AUROC 0.50 on all three; length AUROC 0.40 NQ (95% CI 0.27–0.53), 0.49 HotpotQA (0.36–0.61), 0.50 MS MARCO (0.45–0.56). All CIs include 0.5. Matched clean passages: NQ 36/100, HotpotQA 63/100, MS MARCO 355/500 → NQ's shortcut-controlled results have wide CIs; say so.
6. **Report the length-only baseline in the main results table**, next to starts-with-query.
7. **MS MARCO caution:** length separates the classes better there, so any MS MARCO generalisation claim is checked against the length-only baseline on the same rows.

**Training augmentation = prefix and length together**

8. In **train rows only** (seed 42): strip the prefix from ~50% of poison; prepend the question to ~50% of clean; truncate ~50% of clean by **sentence-end matching against the actual training poison lengths after the prefix step** (a mixture: about half ~42 words with prefix, half ~28 without), final length prefix included. Not the #5 distribution, which is prefix-removed only and is for the test-time pairing. Stored as `train_aug`.
   - *Build result:* 896 of the 1,200 clean passages drawn for truncation matched. On DistilBERT's unique training passages, augmentation moves starts-with-query AUROC 1.00 → 0.50, length 0.74 → 0.68, full stop 0.68 → 0.63. The prefix cue is removed; length and full stop are only reduced. State that in the report.
   - **Both detectors train on the same `train_aug` rows.** DistilBERT uses them as unique passages plus ranks 6–20. The feature scorer uses them as group rows of the train questions (augmented passages placed in their groups, which needs Contriever embeddings for them), and **its negatives include the clean passages inside attacked (n=1) groups**, not only n=0 groups.
   - **Headline detector = augmented** (prefix + length), **fixed now**. **Ablation = no augmentation.** Validation is never augmented, so it cannot show which version resists shortcuts (it would likely favour the no-augmentation model); it is used **only to set the threshold**.
   - Consequence, written in the report: for the headline detector, `prefix_removed` and `clean_truncated` are no longer surprise tests. The **no-augmentation ablation** carries the "did it learn the shortcut?" result, and the gap between the two is the finding.
   - **Val and test are never augmented.**

**TrustRAG-style baseline — use TrustRAG's own removal rule**

9. TrustRAG Stage 1 (Zhou et al., App. B.2): K-means, k=2; thresholds **cosine 0.85** and **ROUGE-L 0.25**. It removes a cluster only when it is a tight, high-overlap group, and **deliberately does not filter a single malicious passage paired with clean ones**; that case goes to an LLM-based Stage 2.
   - **Our version:** K-means (k=2) on L2-normalised Contriever embeddings of each top-5. A cluster qualifies **only if** it has ≥2 passages **and** its mean pairwise cosine (within the cluster) ≥ 0.85 **and** its mean pairwise ROUGE-L (within the cluster) ≥ 0.25. If both clusters qualify, remove only the one with the higher mean cosine. If neither qualifies, remove nothing. (Verified wording, Sec. 5.1: "the K-means filtering strategy will randomly remove the group with higher similarity." That sentence describes K-means *without* N-gram preservation, i.e. a failure case, so our tie rule is inferred from it, not stated by the paper as its rule. Quote it in the report only in that sense.)
   - **Say in the report:** the paper (App. B.2) gives only the two threshold values. It does not say between which passages each is computed, which side triggers removal, how they combine, or in what order, and it has no pseudocode. All of those are our choices. Its encoders also differ from Contriever, and we omit Stage 2. Call it "a simplified TrustRAG Stage-1 baseline".
   - **Calibration check, no tuning:** report the share of clean val contexts (n=0) where it removes anything.
   - For Intro/RW: TrustRAG's own design sends the single-passage case to an extra LLM pass. That supports our cost argument.

**Statistics**

10. **DistilBERT: 3 seeds per configuration — seeds 42, 43, 44** (augmented and not augmented). Each seed is a separate model with its **own threshold, frozen on val**.
    - **Tables show mean ± sd across seeds; bootstrap CIs are for seed 42, the end-to-end model.**
    - **Thresholds (all detectors and baselines):** the 5% and 1% FPR thresholds are set on the **val n=0 clean passages**, then frozen for every test set, MS MARCO and every variant. Report every detection metric as **mean ± sd across the 3 seeds**. The headline configuration is already fixed (#8); val chooses nothing except thresholds.
    - **End-to-end runs use the seed-42 model only**, decided now, so generation cost doesn't triple.
11. **Paired comparison, per seed:** for each DistilBERT seed, bootstrap the **difference** in AUROC (DistilBERT seed − feature scorer) on the same rows, resampling questions within dataset (2,000, seed 42). **Report all three** 95% CIs and whether each excludes 0.

**Fixed before the first detector scores anything** (one line each in `eval_protocol.md`)

12. **Baseline direction.** Each single-feature baseline (length, retrieval score) is oriented once on **train rows**: keep the direction whose train AUROC ≥ 0.5, then freeze it for val, test, MS MARCO and all variants. Expected: shorter = poisoned; higher retrieval score = poisoned. A variant's retrieval score is its Contriever score against the question, recomputed for the changed text. Starts-with-query is binary (starts with the question = poisoned).
13. **Ranks 6–20 are augmented too.** They are three-quarters of DistilBERT's clean training passages, so they get the same clean augmentation (question prepended to ~50%, ~50% truncated, independent draws, seed 42). Otherwise prefix and length would still carry label information through them.
14. **Feature scorer training groups: n=0 and the five n=1 groups per train question. n=5 is deliberately excluded:** its groups are almost all poison, so their group features are degenerate, and the headline condition is n=1. Feature-scorer results at n=5 are therefore out of its training distribution; say so when reporting them.

**Honesty log**

15. Add to `decisions.md`: *"Passage-length statistics, including for the held-out MS MARCO set, were computed before the protocol was frozen and used only to design controls. No detector, feature or threshold was fitted or tuned on MS MARCO."*

Write each change as one line in `decisions.md`, then commit `eval_protocol.md`. After that the protocol is frozen.

---

## 4. Final experiment

**Detectors:** DistilBERT on `[query, passage]` (primary, redundancy-free; headline = augmented, ablation = not augmented) · feature scorer + logistic regression (rival, uses the retrieved set).
**Baselines (same harness):** starts-with-query · **ends with a full stop** (added 17 Oct; NQ's clean passages stop mid-sentence 57–62% of the time, poison never does) · passage length · retrieval score · simplified TrustRAG Stage-1.

**Rule:** AUROC needs both classes, so it is computed **only on the pairings below**. Single-class sets also get the rate that fits their label at the frozen threshold (5%, plus 1%): **FPR for clean sets, TPR for poison sets**.
**Clean side of "Original":** clean passages from n=0 groups and from inside attacked groups. FPR is reported for each source separately (§3 #1).

| Evaluation | Positives (1) | Negatives (0) | Metrics | Read as |
|---|---|---|---|---|
| **Original** | original poison | original clean top-5 | AUROC; TPR and FPR at threshold | In-distribution reference |
| Prefix, poison side | `prefix_removed` | original clean | AUROC; **TPR** on `prefix_removed` | Drop in TPR vs Original → relied on the prefix. A high TPR here may still come from length (prefix-removed poison is even shorter), so read it next to the shortcut-controlled row |
| Prefix, clean side | original poison | `clean_prefixed` | AUROC; **FPR** on `clean_prefixed` | Rise in FPR vs Original clean → relied on the prefix |
| LLM style | original poison | `rewritten_clean` | AUROC; **FPR** on `rewritten_clean` | Rise in FPR vs Original clean → relied on LLM style. (The AUROC still contains prefix/length cues; read it as a change from Original.) |
| **Shortcut-controlled** | `prefix_removed` | `clean_truncated` | AUROC; TPR and FPR at threshold | Neither prefix nor length helps. The most honest single number |
| Unseen attack | `injection` (12–20) | clean top-5 of the same test questions | AUROC (wide CI, say so); **TPR** on `injection` | Generalisation to indirect prompt injection |

Every evaluation is computed **separately for NQ+HotpotQA test and for MS MARCO** (never trained or tuned on).

**End-to-end (n = 0, 1, 5; n=1 is the headline) — test questions only.** NQ and HotpotQA train/val questions trained the detectors and set the thresholds, so they are excluded.
- **Questions:** 40 NQ+HotpotQA test + 100 MS MARCO = **140**. Defences: DistilBERT (seed 42), feature scorer, TrustRAG Stage-1 → 140 × 3 × 3 ≈ **1,260 generations**, filter threshold 5% FPR from val.
- **Undefended reference:** recomputed from Step 1's saved records **on the same 140 questions** (no new generation), so every defence is compared on identical questions. Don't compare against Step 1's all-question numbers.
- **Metrics:** ASR (substring and whole-word, overall and given retrieval), clean F1 retention at n=0, clean passages removed, latency per passage.
- **Limitation to state:** with 40 NQ+HotpotQA test questions, ASR CIs are wide (about ±0.13). MS MARCO's 100 questions give tighter ones.

**Where variant passages sit in a group** (the feature scorer scores relative to the group):
- Each variant passage **replaces its original in the same group, at the same position**.
- **`prefix_removed`:** replaces its poison in each of the five n=1 detection groups (and in n=5 groups); clean passages unchanged.
- **`clean_prefixed` and `rewritten_clean` — one clean passage at a time:** for each clean passage in an n=0 group, a copy of that group with only that passage replaced. This tests the passage-level shortcut without shifting the group's features. Paired against original poison in n=1 groups for AUROC.
- **`clean_truncated` — all clean passages in the group:** used only in the **shortcut-controlled pairing**. Its groups are n=0 groups with every clean passage truncated (for FPR), and n=1 groups with prefix-removed poison j and every clean passage truncated (for TPR and AUROC). The whole group is changed because that pairing needs neither prefix nor length to separate the classes anywhere in the group.
- **`injection`:** replaces the rank-5 passage in **n=0 groups**.
- Changed passages need new Contriever embeddings (Aleena, Sunday, already planned).
**Sensitivity check:** headline numbers with hand-check-flagged questions excluded. Coverage: every NQ and HotpotQA test question was checked; MS MARCO only 30 of 100. State that in the report, so the MS MARCO sensitivity check covers only its checked questions. Flagged test questions to exclude: NQ test442 and test53; HotpotQA none; MS MARCO 156641, 416509, 577564.
**Hand-check tie-break (same for every dataset):** if the attacker answer is false under every reasonable reading of the question, the verdict is `wrong`, with any ambiguity in the note. `ambiguous_question` is used only when some reading makes the attacker answer defensible. (test347 is a train question, so it doesn't affect test results.)
**Labels are by origin, not by hand-check:** an injected passage is label 1 even when its question was flagged. Flags only change the ASR sensitivity check, never detector labels.

**Sizes (give both counts in the report):**
- **Training — unique passages, no duplicates (DistilBERT):** per train question, 5 poison + clean top-5 + ranks 6–20. 120 questions → 600 poison, 600 clean top-5, up to 1,800 extra negatives.
- **Feature scorer trains on group rows of train questions; ranks 6–20 are DistilBERT-only** (they aren't in any group, so group features can't be computed for them).
- **Class imbalance, fixed now:** DistilBERT trains on ~600 poison vs ~2,400 clean (about 1:4), so it uses a **class-weighted loss** (weights inversely proportional to class frequency in the training set). Decided before results so it isn't changed afterwards. AUROC is unaffected; training behaviour and the threshold are.
- **Evaluation — one row per passage per group:** the same clean passage appears in the n=0 group and again in the n=1 group, and poison #1 appears at n=1 and n=5, so evaluation rows outnumber unique passages. Unique passages: val 40 q (200 poison / 200 clean top-5), test 40 q (200/200), MS MARCO 100 q (500/500). With five n=1 detection groups per question, n=1 detection has up to 200 in-distribution and 500 MS MARCO positives (minus any poison that would not be retrieved). Report the unique-passage count **and** the row count per evaluation (the build script prints both).

---

## 5. Figures and tables (decide now, so outputs are saved in the right shape)

| # | What | Source |
|---|---|---|
| Fig 1 | Pipeline: retrieve → trust score → filter → generate, showing what each detector looks at | Draw once |
| Fig 2 | ASR, retrieval success, clean accuracy vs n poisoned (all 3 datasets) | Step 1 (exists) |
| Fig 3 | **Shortcut figure.** Left: AUROC per evaluation in §4 (x) for each detector and baseline, with CIs. Right: rate at the frozen threshold (TPR for poison-only sets, FPR for clean-only sets), with the Original rate as a reference line | Detection results |
| Fig 4 | End-to-end: ASR at n=1 / n=5 and clean F1 retention at n=0, per defence | End-to-end runs |
| Table 1 | Main detection table: AUROC, TPR@5%, TPR@1% per method, for Original, Shortcut-controlled and MS MARCO; augmented vs not augmented | Detection results |
| Table 2 | Design decisions (compact `decisions.md`) | `decisions.md` |

All figures come from **one script** that reads saved outputs, so nothing is hand-copied.

---

## 6. Day by day

| Day | Samana | Aleena | Writing |
|---|---|---|---|
| **Sat 17** | ~~Commit `rag_core.py` + notebook fix~~ (done). ~~Build `data/passages.jsonl`~~ (done: `02_build_dataset.ipynb`, 29,252 rows, all protocol checks pass; re-run when injection / rewrites arrive) | Morning: settle §3 together, commit `eval_protocol.md`. ~~Re-judge 4 NQ + test53~~ (done). Then write the 12–20 injection passages; start feature computation | — |
| **Sun 18** | `rewritten_clean` on Colab (~45 min) + spot-check 20. DistilBERT: augmented and not augmented, 3 seeds each (~5 min/run) | Contriever embeddings for the test variants **and the train_aug passages**. Feature scorer (LR), TrustRAG Stage-1 (with its removal rule), length / retrieval-score / starts-with-query baselines | Aleena: finish Related Work at 1 page, regrouped, DOIs |
| **Mon 19** ✅ | **CHECKPOINT: both detectors trained, thresholds frozen on val.** Score all evaluations with DistilBERT | Shared eval harness: pairings, rates at threshold, CIs, paired difference | Samana: draft **Methodology** (design only, needs no results) |
| **Tue 20** | End-to-end runs, n=0/1/5, DistilBERT + TrustRAG Stage-1 (own Colab) | End-to-end runs, feature scorer (own Colab). **Failure analysis:** 20 misses + 20 false alarms per detector, categorised | Aleena: Introduction rewrite (paragraphs, RQ, hypothesis, contributions) |
| **Wed 21** ✅ | **EXPERIMENTS FROZEN (evening).** Figure/table script | Sensitivity check, latency | Both: start Results |
| **Thu 22** | Results: detection + shortcut tests | Results: end-to-end + failure analysis | Conclusion, limitations, future work. Slide outline |
| **Fri 23** ✅ | **Full draft within page limit.** Update abstract with findings | References (DOIs), contribution breakdown | Swap and read each other's sections |
| **Sat 24** | Polish, page count, figure quality | Polish | **Record presentation** |
| **Sun 25** | Buffer → submit PDF + recording by midday | | |

**Saturday has no slack.** If it slips, **move `rewritten_clean` to Mon/Tue first**. It is independent of everything else and already near the bottom of the cut list.

---

## 7. Page budget (IEEE two-column)

Abstract 150–200 words · Introduction 1–1.25 · Related Work ≤1 · **Method ~3.5** · **Experiments & Results ~3.5** · Conclusion & limitations ~1 · References ≤1 (≥10, DOIs) · Contribution breakdown (short).

## 8. Presentation split (4–5 min each, each presents their own work)

- **Samana:** the problem and why it matters → attack baseline (Fig 2) → DistilBERT → shortcut tests (Fig 3).
- **Aleena:** existing defences and the gap → feature scorer and TrustRAG-style baseline → end-to-end results (Fig 4) → failure analysis, conclusion, future work.

## 9. If we fall behind — cut in this order

1. Latency measurement
2. 3 seeds → 1 seed (say so)
3. End-to-end on MS MARCO (keep MS MARCO detection)
4. `rewritten_clean`

**Never cut:** prefix_removed, clean_prefixed, clean_truncated, the shortcut-controlled pairing, the no-augmentation ablation, starts-with-query and length baselines, n=1 end-to-end, TrustRAG Stage-1 at n=1, failure analysis, the paired comparison.

## 10. What to expect, and how to write it ("we test whether…")

- DistilBERT near-perfect on Original → expected; the controls are the real result.
- Starts-with-query and length baselines strong on Original → shows how easy the shortcuts are, which is what makes the controls meaningful.
- No-augmentation DistilBERT drops on the shortcut-controlled pairing → a finding ("relies partly on surface cues"); the augmented version shows how much of that is fixable.
- TrustRAG Stage-1 removes little at n=1 → consistent with its design, which defers single passages to an LLM pass. Report whatever happens at n=1 and n=5.
- Redundancy features weak at n=1 → tests our motivating claim; report whatever happens.
