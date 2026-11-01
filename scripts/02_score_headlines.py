"""Score every headline with the fine-tuned model from step 1.

Headlines come from FNSPID (Dong et al. 2024), the Nasdaq 100 subset published at
huggingface.co/datasets/benstaf/FNSPID-nasdaq-100-post2019-1newsperrow. Only the
date, ticker and model probabilities are written to data/, not the text.
"""
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from newsalpha import sentiment as S  # noqa: E402


def load_headlines() -> pd.DataFrame:
    path = hf_hub_download("benstaf/FNSPID-nasdaq-100-post2019-1newsperrow",
                           "data/train-00000-of-00001.parquet", repo_type="dataset")
    d = pd.read_parquet(path, columns=["Date", "Article_title", "Stock_symbol"])
    d.columns = ["date", "title", "ticker"]
    d["date"] = pd.to_datetime(d["date"]).dt.tz_convert(None).dt.normalize()
    d = d.dropna(subset=["title"]).drop_duplicates(["title", "ticker"])
    return d[d["date"] < "2024-01-01"].reset_index(drop=True)


def main():
    news = load_headlines()
    model_dir = ROOT / "models" / "finbert-finetuned"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    probs = S.predict_proba(model, tok, news["title"].tolist(), batch_size=256)
    out = news[["date", "ticker"]].copy()
    for i, name in enumerate(S.LABELS):
        out[f"p_{name}"] = probs[:, i].round(4)
    out.to_csv(ROOT / "data" / "headline_scores.csv.gz", index=False)
    print(out.shape, out[[f"p_{n}" for n in S.LABELS]].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
