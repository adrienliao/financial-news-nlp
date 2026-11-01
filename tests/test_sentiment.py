import numpy as np

from newsalpha import sentiment as S


def test_tfidf_baseline_learns_obvious_words():
    texts = ["shares plunge after weak guidance", "stock soars on record profit",
             "company to hold annual meeting", "shares tumble as sales miss",
             "stock jumps after earnings beat", "board announces meeting date"] * 10
    labels = [0, 1, 2, 0, 1, 2] * 10
    clf = S.tfidf_logreg().fit(texts, labels)
    pred = clf.predict(["shares plunge on weak sales", "stock soars after record earnings"])
    assert list(pred) == [0, 1]


def test_finbert_reordering_is_a_permutation():
    assert sorted(S.FINBERT_TO_OURS) == [0, 1, 2]
    probs = np.array([[0.7, 0.2, 0.1]])  # FinBERT: positive, negative, neutral
    assert np.allclose(probs[:, S.FINBERT_TO_OURS], [[0.2, 0.7, 0.1]])
