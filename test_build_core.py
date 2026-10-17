"""Tests for build_core.py. Run: python3 test_build_core.py [path/to/rag_core.py]"""
import sys

import build_core as bc


def test_helpers():
    assert bc.strip_prefix("who won.X won it.", "who won") == "X won it."
    assert bc.add_prefix("Paris is big.", "what city") == "what city.Paris is big."
    try:
        bc.strip_prefix("no prefix here", "who won"); raise AssertionError("should raise")
    except ValueError:
        pass
    # sentence-end matching: only cuts at sentence ends with a length the poison has
    import numpy as np
    from collections import Counter
    txt2 = "A b c. D e f g. H i j k l. M n."              # sentence ends at 3, 7, 12, 14 words
    rng = np.random.default_rng(0)
    assert bc.match_cut(txt2, Counter({7: 5}), rng) == "A b c. D e f g."
    assert bc.match_cut(txt2, Counter({5: 5}), rng) is None              # no sentence end at 5
    assert bc.match_cut(txt2, Counter({10: 1}), rng, offset=3) == "A b c. D e f g."   # 7 + 3 = 10
    outs = {bc.match_cut(txt2, Counter({3: 1, 12: 1000}), np.random.default_rng(s)) for s in range(50)}
    assert "A b c. D e f g. H i j k l." in outs                            # weighted towards common lengths
    assert bc.match_cut("", Counter({3: 1}), rng) is None
    assert bc.ends_with_sentence_punct("It ended.") and bc.ends_with_sentence_punct('He said "yes."')
    assert not bc.ends_with_sentence_punct("cut mid") and not bc.ends_with_sentence_punct("")
    # auroc
    assert bc.auroc([3, 4], [1, 2]) == 1.0 and bc.auroc([1], [1]) == 0.5 and bc.auroc([1, 2], [3, 4]) == 0.0
    # groups: tie rule (clean wins), n=1 group excluded when poison not retrieved
    hits = [(f"d{i}", 10.0 - i) for i in range(20)]
    G = bc.build_groups(hits, [100.0, 6.0, 5.0, 0.0, 9.5])
    assert G["n1"][1][0]["id"] == "adv0"
    assert G["n1"][2] is None      # 6.0 ties d4 (5th clean) -> clean wins -> not retrieved
    assert G["n1"][3] is None and G["n1"][4] is None
    assert [t["id"] for t in G["n1"][5]][:2] == ["d0", "adv4"]
    print("helpers: ok")


def test_merge_matches_rag_core(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("rag_core", path)
    rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
    import random
    rnd = random.Random(0)
    for _ in range(2000):
        hits = sorted([(f"d{i}", rnd.choice([rnd.random(), 0.5])) for i in range(8)], key=lambda x: -x[1])
        adv = [(f"adv{j}", rnd.choice([rnd.random(), 0.5])) for j in range(rnd.randint(0, 5))]
        assert bc.merge_topk(hits, adv, 5) == rc.merge_topk(hits, adv, 5)
    assert bc.build_poisoned_passages("q", ["a", "b"], 2) == rc.build_poisoned_passages("q", ["a", "b"], 2)
    print("merge_topk / build_poisoned_passages identical to rag_core: ok")


if __name__ == "__main__":
    test_helpers()
    if len(sys.argv) > 1:
        test_merge_matches_rag_core(sys.argv[1])
