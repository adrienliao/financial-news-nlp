"""Compare sentiment classifiers on labelled financial headlines and save the best one.

Data: zeroshot/twitter-financial-news-sentiment (about 9.5k train, 2.4k validation
headlines labelled bearish / bullish / neutral). The validation split is used as
the test set; nothing is tuned on it.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from newsalpha import sentiment as S  # noqa: E402

MODEL_DIR = ROOT / "models" / "finbert-finetuned"


def scores(y, probs):
    pred = probs.argmax(1)
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "confusion": confusion_matrix(y, pred).tolist()}


def main():
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    Xtr, ytr = ds["train"]["text"], np.array(ds["train"]["label"])
    Xte, yte = ds["validation"]["text"], np.array(ds["validation"]["label"])
    print(f"train {len(Xtr)}, test {len(Xte)}, class shares {np.bincount(ytr) / len(ytr)}")
    res = {}

    t = time.time()
    clf = S.tfidf_logreg().fit(Xtr, ytr)
    res["tfidf_logreg"] = {**scores(yte, clf.predict_proba(Xte)), "seconds": time.time() - t}

    t = time.time()
    fb, tok = S.load_finbert()
    res["finbert_zero_shot"] = {**scores(yte, S.predict_proba(fb, tok, Xte, reorder=S.FINBERT_TO_OURS)),
                                "seconds": time.time() - t}

    for name, base in [("distilroberta_finetuned", "distilroberta-base"), ("finbert_finetuned", S.FINBERT)]:
        t = time.time()
        model, tok = S.fine_tune(base, Xtr, ytr, epochs=2)
        res[name] = {**scores(yte, S.predict_proba(model, tok, Xte)), "seconds": time.time() - t}
        if name == "finbert_finetuned":
            model.save_pretrained(MODEL_DIR)
            tok.save_pretrained(MODEL_DIR)

    for k, v in res.items():
        print(f"{k:24s} accuracy {v['accuracy']:.3f}  macro F1 {v['macro_f1']:.3f}  ({v['seconds']:.0f}s)")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "sentiment_benchmark.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
