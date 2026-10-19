"""Shared evaluation for all detectors and baselines (eval_protocol.md section 7).

Every detector writes one score per row of data/passages.jsonl (higher = more likely
poisoned). This module turns rows + scores into the protocol's pairings, rates at frozen
thresholds, bootstrap CIs (resampling questions within each dataset) and the paired
AUROC difference. Pure NumPy, no GPU.
"""
import numpy as np

# dataset groups reported separately (section 4)
GROUPS = {"in_dist": ("nq", "hotpotqa"), "msmarco": ("msmarco",)}


# ---------------------------------------------------------------- metrics

def auroc(pos, neg):
    """AUROC of score(pos) > score(neg), ties counted 0.5 (Mann-Whitney)."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    ranks = (np.bincount(inv, weights=ranks) / counts)[inv]
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def threshold_at_fpr(neg_scores, fpr):
    """Threshold t such that flagging score > t gives FPR <= fpr on these negatives
    (section 7: set on val n=0 clean passages, then frozen)."""
    s = np.sort(np.asarray(neg_scores, float))[::-1]
    k = int(np.floor(fpr * len(s)))          # number of false positives allowed
    return float(s[k]) if k < len(s) else float(s[-1]) - 1e-12


def rate(scores, t):
    """Share of scores flagged (score > t): TPR on positives, FPR on negatives."""
    s = np.asarray(scores, float)
    return float((s > t).mean()) if s.size else float("nan")


# ---------------------------------------------------------------- pairings (section 7)

def _is(r, **kw):
    return all(r.get(k) == v for k, v in kw.items())


def pairing(rows, name):
    """Return (positive_rows, negative_rows) for one protocol pairing.
    Positives at n=1 use the five n=1 detection groups (headline condition)."""
    orig_pos = [r for r in rows if _is(r, group_variant="original", n_poison=1, label=1)]
    orig_neg = [r for r in rows if _is(r, group_variant="original", n_poison=0, label=0)]
    if name == "original":
        return orig_pos, orig_neg
    if name == "original_n5":
        return [r for r in rows if _is(r, group_variant="original", n_poison=5, label=1)], orig_neg
    if name == "prefix_poison_side":
        return [r for r in rows if _is(r, group_variant="prefix_removed", n_poison=1, variant="prefix_removed")], orig_neg
    if name == "prefix_clean_side":
        return orig_pos, [r for r in rows if _is(r, group_variant="clean_prefixed", variant="clean_prefixed")]
    if name == "llm_style":
        return orig_pos, [r for r in rows if _is(r, group_variant="rewritten_clean", variant="rewritten_clean")]
    if name == "shortcut_controlled":
        return ([r for r in rows if _is(r, group_variant="shortcut_controlled", n_poison=1, variant="prefix_removed")],
                [r for r in rows if _is(r, group_variant="shortcut_controlled", n_poison=0, variant="clean_truncated")])
    if name == "unseen_attack":
        pos = [r for r in rows if _is(r, group_variant="injection", variant="injection")]
        qs = {(r["dataset"], r["qid"]) for r in pos}
        return pos, [r for r in orig_neg if (r["dataset"], r["qid"]) in qs]
    if name == "collateral_n1":       # FPR only: clean passages inside attacked n=1 groups
        return [], [r for r in rows if _is(r, group_variant="original", n_poison=1, label=0)]
    raise ValueError(name)


PAIRINGS = ["original", "original_n5", "prefix_poison_side", "prefix_clean_side", "llm_style",
            "shortcut_controlled", "unseen_attack", "collateral_n1"]


# ---------------------------------------------------------------- bootstrap by question

def _by_question(rs, score_key):
    out = {}
    for r in rs:
        out.setdefault((r["dataset"], r["qid"]), []).append(r[score_key])
    return out


def _resample_keys(keys, rng):
    """Resample questions within each dataset (so one dataset cannot dominate)."""
    by_ds = {}
    for k in keys:
        by_ds.setdefault(k[0], []).append(k)
    out = []
    for ds in sorted(by_ds):
        ks = by_ds[ds]
        out += [ks[i] for i in rng.integers(0, len(ks), len(ks))]
    return out


def evaluate(rows, score_key, thresholds, n_boot=2000, seed=42, pairings=PAIRINGS, groups=GROUPS):
    """Detection metrics for one score column. thresholds: {"fpr5": t5, "fpr1": t1}.
    Returns {group: {pairing: {...}}} with point estimates, bootstrap CIs and counts."""
    res = {}
    test = [r for r in rows if r["split"] == "test"]
    for g, dsets in groups.items():
        sub = [r for r in test if r["dataset"] in dsets]
        res[g] = {}
        for name in pairings:
            pos, neg = pairing(sub, name)
            if not pos and not neg:
                continue
            P, N = _by_question(pos, score_key), _by_question(neg, score_key)
            keys = sorted(set(P) | set(N))
            ps = [s for k in keys for s in P.get(k, [])]
            ns = [s for k in keys for s in N.get(k, [])]
            out = {"n_pos": len(ps), "n_neg": len(ns), "n_questions": len(keys)}
            point = {"auroc": auroc(ps, ns)}
            for tk, t in thresholds.items():
                point[f"tpr_{tk}"] = rate(ps, t); point[f"fpr_{tk}"] = rate(ns, t)
            rng = np.random.default_rng(seed); boots = {k: [] for k in point}
            for _ in range(n_boot):
                ks = _resample_keys(keys, rng)
                bp = [s for k in ks for s in P.get(k, [])]; bn = [s for k in ks for s in N.get(k, [])]
                boots["auroc"].append(auroc(bp, bn))
                for tk, t in thresholds.items():
                    boots[f"tpr_{tk}"].append(rate(bp, t)); boots[f"fpr_{tk}"].append(rate(bn, t))
            for k, v in point.items():
                b = np.asarray(boots[k], float); b = b[~np.isnan(b)]
                out[k] = {"value": None if np.isnan(v) else round(v, 4),
                          "ci95": [round(float(x), 4) for x in np.quantile(b, [0.025, 0.975])] if b.size else None}
            res[g][name] = out
    return res


def paired_auroc_difference(rows, key_a, key_b, name, group_dsets, n_boot=2000, seed=42):
    """Section 7: bootstrap the difference AUROC(a) - AUROC(b) on the same rows,
    resampling questions within dataset. Returns value, 95% CI and whether it excludes 0."""
    test = [r for r in rows if r["split"] == "test" and r["dataset"] in group_dsets]
    pos, neg = pairing(test, name)
    keys = sorted({(r["dataset"], r["qid"]) for r in pos + neg})
    Pa, Na, Pb, Nb = (_by_question(x, k) for x, k in ((pos, key_a), (neg, key_a), (pos, key_b), (neg, key_b)))
    def diff(ks):
        a = auroc([s for k in ks for s in Pa.get(k, [])], [s for k in ks for s in Na.get(k, [])])
        b = auroc([s for k in ks for s in Pb.get(k, [])], [s for k in ks for s in Nb.get(k, [])])
        return a - b
    point = diff(keys); rng = np.random.default_rng(seed)
    boots = np.array([diff(_resample_keys(keys, rng)) for _ in range(n_boot)])
    boots = boots[~np.isnan(boots)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"diff": round(float(point), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def val_clean_n0(rows, score_key):
    """Section 7: thresholds are set on val n=0 clean passages (NQ + HotpotQA val)."""
    return [r[score_key] for r in rows
            if r["split"] == "val" and _is(r, group_variant="original", n_poison=0, label=0)]
