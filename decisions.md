# Decisions log

One line per decision: what we chose and why. This becomes the Methodology section.

| Date | Decision | Reason |
|---|---|---|
| 2026-10-14 | Use PoisonedRAG's released NQ targets (100 questions, correct + attacker answers, 5 poisoned texts each) | Reuses the published attack, so our attack success rate is comparable with Zou et al. |
| 2026-10-14 | Build each poisoned passage as question + "." + released text | This is exactly how PoisonedRAG's black-box attack constructs them (`src/attack.py`) |
| 2026-10-14 | Use Contriever instead of bge-small (changes the spec) | It produced PoisonedRAG's stored rankings, which lets us use the full 2.68M-passage NQ corpus without re-indexing |
| 2026-10-14 | Simulate injection by merging poisoned passages with the clean top-k | Exact for exact inner-product search: adding passages can only push clean ones down |
| 2026-10-14 | Generator: Qwen2.5-3B-Instruct, fp16, greedy decoding | Open, fits a free T4, and greedy decoding makes runs repeatable |
| 2026-10-14 | Attack success = response contains the attacker's answer (PoisonedRAG's string test) | Same criterion as the paper |
| 2026-10-14 | Clean accuracy = token F1 vs correct answer, plus "contains correct answer" | F1 is our agreed primary metric; containment allows comparison with PoisonedRAG |
| 2026-10-14 | 60/20/20 train/val/test split by question, seed 42, fixed in Step 1 | Fixed before any detector exists, so no test information can leak into training |
| 2026-10-14 | 95% bootstrap CIs, resampling questions | The question is the unit of analysis |
| 2026-10-14 | Baseline uses PoisonedRAG's plain prompt, with no "treat documents as untrusted" instruction | That instruction is itself a defence and would contaminate the undefended baseline; kept as an optional skeptical-prompting baseline instead |
| 2026-10-14 | Report attack success both overall and conditional on the poisoned passage being retrieved | Separates retrieval failures ("not retrieved") from generation failures ("retrieved but not believed") |
| 2026-10-14 | Check clean retrieval before generation: Hit@5 against gold passages, and whether the correct answer appears in the top 5 | Sets the ceiling for clean accuracy and shows whether low accuracy comes from retrieval or the generator |
| 2026-10-14 | Pin transformers to >=4.45,<5 | Prevents a library update changing behaviour mid-project |
| 2026-10-14 | Use PoisonedRAG's released NQ, HotpotQA and MS MARCO sets instead of generating our own poisoned passages | 300 questions in one consistent format from a single attack source, without LLM generation time |
| 2026-10-14 | NQ and HotpotQA split 60/20/20 by question; MS MARCO held out entirely as a test-only dataset | Pools 40 in-distribution test questions and adds 100 from a dataset the detectors never see in training |
| 2026-10-14 | MS MARCO uses its train-split questions and ground truth | This is the split PoisonedRAG used (evaluate_beir.py) |
| 2026-10-14 | Hand-check ~30 attacker answers per dataset for "the wrong answer is actually valid" | The released attacker answers were LLM-generated and may occasionally be correct |
| 2026-10-14 | Add a prefix-removed test set (poisoned passages without the leading question) | Every poisoned passage starts with the question, so a detector could learn that shortcut instead of anything about poisoning |
