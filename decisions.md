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
