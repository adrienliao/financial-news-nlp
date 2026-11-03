"""Three ways to classify a financial headline as bearish, bullish or neutral.

1. TF-IDF on word and character n-grams + logistic regression: the baseline
   every NLP model has to beat.
2. FinBERT (Araci 2019) used as is, without training on our labels.
3. A pretrained transformer fine-tuned on our labels with a plain PyTorch loop.

Label convention everywhere: 0 = bearish, 1 = bullish, 2 = neutral, the
convention of the twitter-financial-news-sentiment dataset.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

LABELS = ["bearish", "bullish", "neutral"]
FINBERT = "ProsusAI/finbert"
# FinBERT outputs (positive, negative, neutral); reorder to (bearish, bullish, neutral)
FINBERT_TO_OURS = [1, 0, 2]


def tfidf_logreg(C: float = 4.0):
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)),
    ])
    return make_pipeline(features, LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))


@torch.no_grad()
def predict_proba(model, tokenizer, texts, batch_size=128, max_length=64, reorder=None) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(list(texts[i:i + batch_size]), padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        out.append(torch.softmax(model(**enc).logits, dim=-1).numpy())
    probs = np.concatenate(out)
    return probs[:, reorder] if reorder is not None else probs


def load_finbert():
    return AutoModelForSequenceClassification.from_pretrained(FINBERT), AutoTokenizer.from_pretrained(FINBERT)


def fine_tune(base: str, train_texts, train_labels, epochs: int = 2, lr: float = 3e-5, batch_size: int = 32,
              max_length: int = 64, seed: int = 0, log_every: int = 50):
    """Standard recipe: AdamW, linear warmup then decay, cross entropy on the [CLS] head."""
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=3, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)},
        ignore_mismatched_sizes=True,
    )
    data = list(zip(train_texts, train_labels))
    loader = DataLoader(data, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed),
                        collate_fn=lambda b: (list(t for t, _ in b), torch.tensor([y for _, y in b])))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(loader)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)

    model.train()
    step, t0 = 0, time.time()
    for _ in range(epochs):
        for texts, y in loader:
            enc = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            loss = model(**enc, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if log_every and step % log_every == 0:
                print(f"  step {step}/{total}  loss {loss.item():.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return model, tok
