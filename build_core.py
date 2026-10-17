"""Step 2: build data/passages.jsonl, following eval_protocol.md (frozen 17 Oct 2026).

Pure Python / NumPy, no GPU, so it can be tested offline. Every rule here is a
rule from eval_protocol.md; section numbers refer to that file.

Columns (protocol section 3, plus two helper columns):
  dataset, qid, split, context_id, n_poison, group_variant, variant, source,
  poison_idx, rank, retrieval_score, question, passage, label, passage_id

  group_variant  the condition of the whole group the row sits in (original,
                 prefix_removed, clean_prefixed, rewritten_clean,
                 shortcut_controlled, injection, train_aug). An unchanged
                 passage inside a variant group keeps variant="original".
  passage_id     one id per (dataset, question, passage), shared by all forms of
                 that passage, so the same passage gets the same augmentation
                 everywhere and DistilBERT can de-duplicate.

Rows with context_id = None are DistilBERT-only training passages that sit in no
group (ranks 6-20, and any poison passage that is never retrieved).
"""
import re

import numpy as np

SEED = 42
TOP_K = 5
EXTRA_RANKS = (6, 20)        # protocol section 5: ranks 6-20 as extra DistilBERT negatives

COLUMNS = ["dataset", "qid", "split", "context_id", "n_poison", "group_variant", "variant",
           "source", "poison_idx", "rank", "retrieval_score", "question", "passage", "label",
           "passage_id"]


# ---------------------------------------------------------------- copied from rag_core.py
# Verbatim copies of rag_core.merge_topk / build_poisoned_passages (frozen 17 Oct), so this
# file runs on its own in Colab. test_build_core.py checks they behave identically.

def merge_topk(clean_hits, adv_hits, k):
    if len(clean_hits) < k:
        raise ValueError(f"need at least k={k} clean hits, got {len(clean_hits)}")
    pool = [{"source": "clean", "id": d, "score": float(s)} for d, s in clean_hits[:k]]
    pool += [{"source": "adv", "id": a, "score": float(s)} for a, s in adv_hits]
    pool.sort(key=lambda r: -r["score"])
    return pool[:k]


def build_poisoned_passages(question, adv_texts, n):
    return [question + "." + t for t in adv_texts[:n]]


# ---------------------------------------------------------------- text helpers

def n_words(s):
    return len(str(s).split())


def strip_prefix(passage, question):
    """prefix_removed: drop the leading 'question.' that PoisonedRAG adds."""
    prefix = question + "."
    if not passage.startswith(prefix):
        raise ValueError("poison passage does not start with its question")
    return passage[len(prefix):]


def add_prefix(text, question):
    """clean_prefixed: same construction as the poison (question + '.' + text, no space)."""
    return question + "." + text


_SENT_END = re.compile(r"[.!?][\"'\)\]]*$")


def ends_with_sentence_punct(text):
    """Baseline 'ends with a full stop' (protocol section 8, added 17 Oct): last token
    ends in . ! or ? (optionally followed by a closing quote or bracket)."""
    w = str(text).split()
    return bool(w) and bool(_SENT_END.search(w[-1]))


def match_cut(text, length_counts, rng, offset=0):
    """Sentence-end matching (protocol sections 5-6, as amended 17 Oct): cut only at a
    sentence end whose final length (cut words + offset) is a length the poison
    actually has; choose among those cut points in proportion to how common that
    length is in the poison. Returns None when no cut point matches (the passage
    is then left unchanged). `offset` = n_words(question) - 1 when the question
    will be prepended afterwards."""
    words = str(text).split()
    ends = [i + 1 for i, w in enumerate(words) if _SENT_END.search(w)]
    cands = [e for e in ends if (e + offset) in length_counts]
    if not cands:
        return None
    p = np.array([length_counts[e + offset] for e in cands], float)
    cut = int(rng.choice(cands, p=p / p.sum()))
    return " ".join(words[:cut])


def auroc(pos, neg):
    """AUROC of score(pos) > score(neg), ties counted as 0.5 (Mann-Whitney)."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


# ---------------------------------------------------------------- groups

def clean_pid(ds, qid, doc_id):
    return f"{ds}:{qid}:doc:{doc_id}"


def poison_pid(ds, qid, idx1):
    return f"{ds}:{qid}:p{idx1}"


def build_groups(clean_hits, adv_scores, k=TOP_K):
    """Section 1. Returns {'n0': top, 'n1': {j: top or None}, 'n5': top} where each
    top is merge_topk output (best first). n=1 detection groups use poison j alone;
    None when poison j would not be retrieved."""
    n0 = merge_topk(clean_hits, [], k)
    n1 = {}
    for j in range(1, 6):
        top = merge_topk(clean_hits, [(f"adv{j - 1}", adv_scores[j - 1])], k)
        n1[j] = top if any(t["source"] == "adv" for t in top) else None
    n5 = merge_topk(clean_hits, [(f"adv{i}", adv_scores[i]) for i in range(5)], k)
    return {"n0": n0, "n1": n1, "n5": n5}


def _base_row(ds, qid, split, question, entry, rank, docs, poison):
    """Original form of one retrieved entry as a row (context fields filled later)."""
    if entry["source"] == "clean":
        return {"source": "clean", "poison_idx": None, "rank": rank,
                "retrieval_score": entry["score"], "passage": docs[entry["id"]]["text"],
                "label": 0, "passage_id": clean_pid(ds, qid, entry["id"])}
    i = int(entry["id"][3:])                       # adv0..adv4 -> poison_idx 1..5
    return {"source": "poison", "poison_idx": i + 1, "rank": rank,
            "retrieval_score": entry["score"], "passage": poison[i],
            "label": 1, "passage_id": poison_pid(ds, qid, i + 1)}


def _finish(row, ds, qid, split, question, context_id, n_poison, group_variant, variant):
    out = {"dataset": ds, "qid": qid, "split": split, "context_id": context_id,
           "n_poison": n_poison, "group_variant": group_variant, "variant": variant,
           "question": question}
    out.update(row)
    return {c: out.get(c) for c in COLUMNS}


# ---------------------------------------------------------------- build

def build_dataset(inputs, injection=None, rewrites=None, seed=SEED):
    """inputs: {dataset: {"targets", "clean_hits", "docs", "adv_scores", "split"}}
         targets[qid]   PoisonedRAG target entry (question, adv_texts, ...)
         clean_hits[q]  [(doc_id, score)] best first, >= 20 entries
         docs[doc_id]   {"title", "text"}
         adv_scores[q]  5 Contriever scores of the prefixed poison passages
         split          {"train": [...], "val": [...], "test": [...]}
       injection: list of {"dataset", "qid", "passage", ...} (test questions only), or None
       rewrites:  {passage_id: rewritten text} for test clean passages, or None
    Returns (rows, report, to_rewrite)."""
    rows, report, to_rewrite = [], {"datasets": {}}, []
    train_units = []            # (passage_id, kind, row-template, question) for augmentation
    groups_all = {}

    for ds in sorted(inputs):
        inp = inputs[ds]
        where = {q: s for s, qs in inp["split"].items() for q in qs}
        rep = {"questions": {}, "n1_groups_built": 0, "n1_poison_not_retrieved": [],
               "clean_truncated_matched": 0, "clean_truncated_unmatched": 0}
        for s in ("train", "val", "test"):
            rep["questions"][s] = len(inp["split"].get(s, []))
        for q in sorted(inp["targets"]):
            t = inp["targets"][q]; question = t["question"]; split = where[q]
            poison = build_poisoned_passages(question, t["adv_texts"], 5)
            hits = inp["clean_hits"][q]
            G = build_groups(hits, inp["adv_scores"][q])
            groups_all[(ds, q)] = G

            def emit(top, ctx, n, gvar, replace=None):
                """Write one group; `replace` maps rank -> (variant, text) for changed passages."""
                for rank, e in enumerate(top, start=1):
                    r = _base_row(ds, q, split, question, e, rank, inp["docs"], poison)
                    variant = "original"
                    if replace and rank in replace:
                        variant, text = replace[rank]
                        r["passage"], r["retrieval_score"] = text, None   # recomputed later (Contriever)
                    rows.append(_finish(r, ds, q, split, question, ctx, n, gvar, variant))

            base = f"{ds}|{q}"
            # --- original groups, every split (section 1)
            emit(G["n0"], f"{base}|n0", 0, "original")
            for j, top in G["n1"].items():
                if top is None:
                    rep["n1_poison_not_retrieved"].append(f"{q}:p{j}"); continue
                rep["n1_groups_built"] += 1
                emit(top, f"{base}|n1|p{j}", 1, "original")
            emit(G["n5"], f"{base}|n5", 5, "original")

            # --- training units (section 5): clean top-5, ranks 6-20, all 5 poison
            if split == "train":
                lo, hi = EXTRA_RANKS
                for rank, (d, sc) in enumerate(hits[:hi], start=1):
                    tmpl = {"source": "clean", "poison_idx": None, "rank": rank,
                            "retrieval_score": float(sc), "passage": inp["docs"][d]["text"],
                            "label": 0, "passage_id": clean_pid(ds, q, d)}
                    train_units.append((ds, q, split, question, "clean", tmpl))
                for i in range(5):
                    tmpl = {"source": "poison", "poison_idx": i + 1, "rank": None,
                            "retrieval_score": float(inp["adv_scores"][q][i]), "passage": poison[i],
                            "label": 1, "passage_id": poison_pid(ds, q, i + 1)}
                    train_units.append((ds, q, split, question, "poison", tmpl))
        report["datasets"][ds] = rep

    # ------------------------------------------------ test variants (section 6)
    for ds in sorted(inputs):
        inp = inputs[ds]; rep = report["datasets"][ds]
        where = {q: s for s, qs in inp["split"].items() for q in qs}
        # clean_truncated: sentence-end matching against the prefix_removed lengths of this
        # dataset (all its poison). One truncated form per clean passage; None = unmatched,
        # which stays unchanged in its group and is not a negative in the pairing.
        from collections import Counter
        pr_counts = Counter(n_words(strip_prefix(p, inp["targets"][q]["question"]))
                            for q in sorted(inp["targets"])
                            for p in build_poisoned_passages(inp["targets"][q]["question"],
                                                             inp["targets"][q]["adv_texts"], 5))
        rng_trunc = np.random.default_rng(seed)
        truncated = {}
        test_qs = [q for q in sorted(inp["targets"]) if where[q] == "test"]
        for q in test_qs:                                   # fixed order -> reproducible
            for e in groups_all[(ds, q)]["n0"]:
                pid = clean_pid(ds, q, e["id"])
                truncated[pid] = match_cut(inp["docs"][e["id"]]["text"], pr_counts, rng_trunc)
                rep["clean_truncated_matched" if truncated[pid] is not None
                    else "clean_truncated_unmatched"] += 1
        for q in test_qs:
            t = inp["targets"][q]; question = t["question"]
            poison = build_poisoned_passages(question, t["adv_texts"], 5)
            G = groups_all[(ds, q)]; base = f"{ds}|{q}"

            def emit(top, ctx, n, gvar, replace):
                for rank, e in enumerate(top, start=1):
                    r = _base_row(ds, q, "test", question, e, rank, inp["docs"], poison)
                    variant = "original"
                    if rank in replace:
                        variant, text = replace[rank]
                        r["passage"], r["retrieval_score"] = text, None
                        if variant == "injection":
                            r.update(source="poison", poison_idx=None, label=1,
                                     passage_id=text_pid)
                    rows.append(_finish(r, ds, q, "test", question, ctx, n, gvar, variant))

            def strip_all(top):
                return {rk: ("prefix_removed", strip_prefix(poison[int(e["id"][3:])], question))
                        for rk, e in enumerate(top, start=1) if e["source"] == "adv"}

            def trunc_all(top):
                """Every matched clean passage in the group is replaced; unmatched ones stay."""
                out = {}
                for rk, e in enumerate(top, start=1):
                    if e["source"] == "clean":
                        cut = truncated[clean_pid(ds, q, e["id"])]
                        if cut is not None:
                            out[rk] = ("clean_truncated", cut)
                return out

            # prefix_removed: replaces its poison in each n=1 group and in the n=5 group
            for j, top in G["n1"].items():
                if top is not None:
                    emit(top, f"{base}|n1|p{j}|prefix_removed", 1, "prefix_removed", strip_all(top))
            emit(G["n5"], f"{base}|n5|prefix_removed", 5, "prefix_removed", strip_all(G["n5"]))

            # clean_prefixed / rewritten_clean: one clean passage at a time, copies of n=0
            for rk, e in enumerate(G["n0"], start=1):
                txt = inp["docs"][e["id"]]["text"]; pid = clean_pid(ds, q, e["id"])
                emit(G["n0"], f"{base}|n0|clean_prefixed|r{rk}", 0, "clean_prefixed",
                     {rk: ("clean_prefixed", add_prefix(txt, question))})
                to_rewrite.append({"passage_id": pid, "dataset": ds, "qid": q,
                                   "question": question, "text": txt})
                if rewrites and pid in rewrites:
                    emit(G["n0"], f"{base}|n0|rewritten_clean|r{rk}", 0, "rewritten_clean",
                         {rk: ("rewritten_clean", rewrites[pid])})

            # shortcut_controlled: all clean truncated; n=0 groups, and n=1 groups with prefix_removed poison
            emit(G["n0"], f"{base}|n0|shortcut_controlled", 0, "shortcut_controlled", trunc_all(G["n0"]))
            for j, top in G["n1"].items():
                if top is not None:
                    rep_map = trunc_all(top); rep_map.update(strip_all(top))
                    emit(top, f"{base}|n1|p{j}|shortcut_controlled", 1, "shortcut_controlled", rep_map)

            # injection: replaces the rank-5 passage in the n=0 group
            for k, item in enumerate([x for x in (injection or [])
                                      if x["dataset"] == ds and x["qid"] == q], start=1):
                text_pid = f"{ds}:{q}:inj{k}"
                emit(G["n0"], f"{base}|n0|injection|{k}", 0, "injection",
                     {TOP_K: ("injection", item["passage"])})

    if injection:
        test_set = {(ds, q) for ds in inputs for q in inputs[ds]["split"].get("test", [])}
        bad = [(x["dataset"], x["qid"]) for x in injection if (x["dataset"], x["qid"]) not in test_set]
        if bad:
            raise ValueError(f"injection passages must be on test questions only: {bad[:5]}")

    # ------------------------------------------------ training augmentation (section 5)
    units = {}                                   # passage_id -> (ds, q, split, question, kind, tmpl)
    for u in train_units:
        units.setdefault(u[5]["passage_id"], u)
    rng = np.random.default_rng(seed)
    poison_ids = sorted(p for p, u in units.items() if u[4] == "poison")
    clean_ids = sorted(p for p, u in units.items() if u[4] == "clean")
    strip_set = set(np.array(poison_ids)[rng.permutation(len(poison_ids))[:len(poison_ids) // 2]])
    aug_text = {}
    for p in poison_ids:
        _, _, _, question, _, tmpl = units[p]
        aug_text[p] = strip_prefix(tmpl["passage"], question) if p in strip_set else tmpl["passage"]
    train_poison_lengths = [n_words(aug_text[p]) for p in poison_ids]   # after the prefix step
    prefix_set = set(np.array(clean_ids)[rng.permutation(len(clean_ids))[:len(clean_ids) // 2]])
    trunc_set = set(np.array(clean_ids)[rng.permutation(len(clean_ids))[:len(clean_ids) // 2]])
    from collections import Counter
    train_len_counts = Counter(train_poison_lengths)
    trunc_matched = 0
    for p in clean_ids:                                  # fixed order -> reproducible
        _, _, _, question, _, tmpl = units[p]
        txt, pre = tmpl["passage"], p in prefix_set
        if p in trunc_set:
            # final length (prefix included) must be a real training-poison length
            cut = match_cut(txt, train_len_counts, rng, offset=(n_words(question) - 1) if pre else 0)
            if cut is not None:
                txt = cut; trunc_matched += 1
        aug_text[p] = add_prefix(txt, question) if pre else txt
    changed = {p for p in aug_text if aug_text[p] != units[p][5]["passage"]}

    # aug groups for train questions: n=0 and the five n=1 groups (feature scorer, section 5)
    grouped = {"original": set(), "train_aug": set()}
    for r in rows:
        if r["split"] == "train" and r["group_variant"] == "original":
            grouped["original"].add(r["passage_id"])
    aug_rows = []
    for (ds, q), G in sorted(groups_all.items()):
        inp = inputs[ds]
        if q not in inp["split"].get("train", []):
            continue
        question = inp["targets"][q]["question"]
        poison = build_poisoned_passages(question, inp["targets"][q]["adv_texts"], 5)
        tops = [(G["n0"], f"{ds}|{q}|n0|train_aug", 0)]
        tops += [(top, f"{ds}|{q}|n1|p{j}|train_aug", 1) for j, top in G["n1"].items() if top is not None]
        for top, ctx, n in tops:
            for rank, e in enumerate(top, start=1):
                r = _base_row(ds, q, "train", question, e, rank, inp["docs"], poison)
                variant = "original"
                if r["passage_id"] in changed:
                    variant = "train_aug"
                    r["passage"], r["retrieval_score"] = aug_text[r["passage_id"]], None
                grouped["train_aug"].add(r["passage_id"])
                aug_rows.append(_finish(r, ds, q, "train", question, ctx, n, "train_aug", variant))
    rows.extend(aug_rows)

    # standalone DistilBERT-only rows: every training unit not in a group of that configuration
    for p in sorted(units):
        ds, q, split, question, kind, tmpl = units[p]
        for gvar in ("original", "train_aug"):
            if p in grouped[gvar]:
                continue
            r = dict(tmpl)
            variant = "original"
            if gvar == "train_aug" and p in changed:
                variant = "train_aug"; r["passage"], r["retrieval_score"] = aug_text[p], None
            rows.append(_finish(r, ds, q, split, question, None, None, gvar, variant))

    report["augmentation"] = {
        "train_poison": len(poison_ids), "poison_prefix_stripped": len(strip_set),
        "train_clean": len(clean_ids), "clean_prefixed": len(prefix_set),
        "clean_drawn_for_truncation": len(trunc_set), "clean_truncation_matched": trunc_matched,
        "clean_drawn_for_both": len(prefix_set & trunc_set),
        "passages_changed": len(changed),
    }
    return rows, report, to_rewrite


# ---------------------------------------------------------------- checks and counts

def check_against_step1(groups_from_build, records):
    """Rebuilt groups must equal what Step 1 actually used: n=1 (poison #1 only) and n=5."""
    bad = []
    for r in records:
        if r["n_poison"] not in (1, 5):
            continue
        ours = [(t["source"], t["id"]) for t in r["topk"]]
        G = groups_from_build[r["qid"]]
        top = G["n1"][1] if r["n_poison"] == 1 else G["n5"]
        if top is None:     # poison #1 not retrieved: Step 1 group is the clean top-5
            top = G["n0"]
        if ours != [(t["source"], t["id"]) for t in top]:
            bad.append((r["qid"], r["n_poison"]))
    return bad


def train_cue_check(rows):
    """How much prefix, length and full-stop cues separate the classes in DistilBERT's
    unique training passages, without vs with augmentation (train data only)."""
    out = {}
    for cfg in ("original", "train_aug"):
        seen = {}
        for r in rows:
            if r["split"] == "train" and r["group_variant"] == cfg:
                seen.setdefault(r["passage_id"], (r["label"], r["passage"], r["question"]))
        pos = [v for v in seen.values() if v[0] == 1]; neg = [v for v in seen.values() if v[0] == 0]
        starts = lambda v: float(v[1].startswith(v[2] + "."))
        out[cfg] = {"n_pos_neg": [len(pos), len(neg)],
                    "starts_with_query_auroc": round(auroc([starts(v) for v in pos], [starts(v) for v in neg]), 3),
                    "length_auroc": round(auroc([-n_words(v[1]) for v in pos], [-n_words(v[1]) for v in neg]), 3),
                    "full_stop_auroc": round(auroc([float(ends_with_sentence_punct(v[1])) for v in pos],
                                                   [float(ends_with_sentence_punct(v[1])) for v in neg]), 3)}
    return out


def summarise_rows(rows):
    """Unique-passage and row counts for the report (protocol: give both)."""
    from collections import Counter
    out = {}
    out["rows_total"] = len(rows)
    out["rows_by_group_variant_n"] = dict(Counter(f"{r['group_variant']}|n={r['n_poison']}" for r in rows))
    out["rows_by_variant_label"] = dict(Counter(f"{r['variant']}|label={r['label']}" for r in rows))
    uniq = {}
    for r in rows:
        if r["group_variant"] != "original":
            continue
        key = (r["dataset"], r["split"], r["label"])
        uniq.setdefault(key, set()).add(r["passage_id"])
    out["unique_passages_original"] = {f"{d}|{s}|label={l}": len(v) for (d, s, l), v in sorted(uniq.items())}
    for cfg in ("original", "train_aug"):
        u = {}
        for r in rows:
            if r["split"] == "train" and r["group_variant"] == cfg:
                u.setdefault(r["label"], set()).add(r["passage_id"])
        out[f"distilbert_train_unique_{cfg}"] = {f"label={k}": len(v) for k, v in sorted(u.items())}
    return out


def length_check(rows):
    """Length-only AUROC (shorter = poisoned) on unique test passages, per dataset:
    Original pairing (for reference) and the shortcut-controlled pairing (should be ~0.5)."""
    out = {}
    for ds in sorted({r["dataset"] for r in rows}):
        def texts(filter_fn):
            seen = {}
            for r in rows:
                if r["dataset"] == ds and r["split"] == "test" and filter_fn(r):
                    seen.setdefault(r["passage_id"], r["passage"])
            return list(seen.values())
        f_op = lambda r: r["group_variant"] == "original" and r["label"] == 1
        f_on = lambda r: r["group_variant"] == "original" and r["label"] == 0
        f_sp = lambda r: r["group_variant"] == "shortcut_controlled" and r["variant"] == "prefix_removed"
        f_sn = lambda r: r["group_variant"] == "shortcut_controlled" and r["variant"] == "clean_truncated"
        op, on, sp, sn = (texts(f) for f in (f_op, f_on, f_sp, f_sn))
        L = lambda xs: [-n_words(t) for t in xs]                      # shorter = poisoned
        E = lambda xs: [float(ends_with_sentence_punct(t)) for t in xs]
        rate = lambda xs: round(float(np.mean(E(xs))), 3) if xs else None
        out[ds] = {"original_pairing": {"length_auroc": round(auroc(L(op), L(on)), 3),
                                        "full_stop_auroc": round(auroc(E(op), E(on)), 3),
                                        "ends_with_full_stop": {"poison": rate(op), "clean": rate(on)},
                                        "n_pos_neg": [len(op), len(on)]},
                   "shortcut_controlled": {"length_auroc": round(auroc(L(sp), L(sn)), 3),
                                           "full_stop_auroc": round(auroc(E(sp), E(sn)), 3),
                                           "ends_with_full_stop": {"poison": rate(sp), "clean": rate(sn)},
                                           "n_pos_neg": [len(sp), len(sn)],
                                           "median_words": {"prefix_removed": float(np.median([n_words(t) for t in sp])) if sp else None,
                                                            "clean_truncated": float(np.median([n_words(t) for t in sn])) if sn else None}}}
    return out


def validate(rows):
    """Protocol compliance checks on the built rows. Returns a list of problems (empty = ok)."""
    from collections import defaultdict
    problems = []
    TEST_ONLY = {"prefix_removed", "clean_prefixed", "rewritten_clean", "shortcut_controlled", "injection"}
    groups = defaultdict(list)
    for r in rows:
        gv, v, sp = r["group_variant"], r["variant"], r["split"]
        if gv in TEST_ONLY and sp != "test":
            problems.append(f"{gv} row outside test: {r['context_id']}")
        if (gv == "train_aug" or v == "train_aug") and sp != "train":
            problems.append(f"train_aug outside train: {r['passage_id']}")
        if r["dataset"] == "msmarco" and sp != "test":
            problems.append(f"MS MARCO row in {sp}")
        if (v != "original") != (r["retrieval_score"] is None):
            problems.append(f"retrieval_score should be None exactly for changed text: {r['passage_id']} {v}")
        if v == "prefix_removed" and (r["label"] != 1 or r["passage"].startswith(r["question"] + ".")):
            problems.append(f"bad prefix_removed: {r['passage_id']}")
        if v == "clean_prefixed" and (r["label"] != 0 or not r["passage"].startswith(r["question"] + ".")):
            problems.append(f"bad clean_prefixed: {r['passage_id']}")
        if v == "clean_truncated" and (r["label"] != 0 or not ends_with_sentence_punct(r["passage"])):
            problems.append(f"bad clean_truncated: {r['passage_id']}")
        if r["label"] != (1 if r["source"] == "poison" else 0):
            problems.append(f"label/source mismatch: {r['passage_id']}")
        if r["context_id"] is not None:
            groups[r["context_id"]].append(r)
    for ctx, g in groups.items():
        ranks = sorted(x["rank"] for x in g)
        if ranks != list(range(1, len(g) + 1)) or len(g) != TOP_K:
            problems.append(f"group {ctx} has ranks {ranks}")
        gv = g[0]["group_variant"]
        changed = [x for x in g if x["variant"] != "original"]
        if gv == "clean_prefixed" and len(changed) != 1:
            problems.append(f"{ctx}: clean_prefixed group must change exactly one passage")
        if gv in ("prefix_removed", "shortcut_controlled") and any(
                x["source"] == "poison" and x["variant"] != "prefix_removed" for x in g):
            problems.append(f"{ctx}: poison not prefix_removed")
        if gv == "shortcut_controlled" and any(
                x["source"] == "clean" and x["variant"] not in ("clean_truncated", "original") for x in g):
            problems.append(f"{ctx}: unexpected clean variant")
        if len({x["split"] for x in g}) != 1 or len({x["qid"] for x in g}) != 1:
            problems.append(f"{ctx}: mixes questions or splits")
    return problems
