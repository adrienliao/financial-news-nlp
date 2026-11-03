"""Do headline sentiment scores move prices, and can you trade on them after the fact?

Reads data/headline_scores.csv.gz and data/close.csv.gz, writes results/event_study.json
and the figures in docs/.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from newsalpha import events  # noqa: E402

BLUE, ORANGE, AQUA, GREY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#d03b3b"
plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#b5b4ae", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.labelcolor": GREY, "xtick.color": GREY, "ytick.color": GREY, "legend.frameon": False,
})
THRESH = 0.5
GROUPS = [("positive", AQUA), ("neutral", GREY), ("negative", RED)]


def group_of(score):
    return np.where(score > THRESH, "positive", np.where(score < -THRESH, "negative", "neutral"))


def sharpe(r):
    r = r[r != 0]
    return float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 1 else float("nan")


def main():
    close = pd.read_csv(ROOT / "data" / "close.csv.gz", index_col="date", parse_dates=True)
    qqq = close.pop("QQQ")
    ar = events.abnormal_returns(close, qqq).loc["2019-01-01":]
    news = pd.read_csv(ROOT / "data" / "headline_scores.csv.gz", parse_dates=["date"])
    news = news[news["ticker"].isin(close.columns)]
    news["score"] = news["p_bullish"] - news["p_bearish"]

    sig = events.daily_signal(news, ar.index)
    sig["group"] = group_of(sig["score"])
    window = events.event_window(sig, ar, -5, 10)
    res = {"n_headlines": int(len(news)), "n_events": int(window.dropna().shape[0]),
           "group_counts": sig["group"].value_counts().to_dict(), "car": {}}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for g, c in GROUPS:
        w = window[sig["group"] == g]
        full = events.car_with_ci(w, -5)
        after = events.car_with_ci(w, 2)
        res["car"][g] = {"day_minus5_to_1": float(full.loc[1, "car"]), "day0_to_1": float(w[[0, 1]].sum(axis=1).mean()),
                         "day2_to_10": float(after.loc[10, "car"]), "day2_to_10_lo": float(after.loc[10, "lo"]),
                         "day2_to_10_hi": float(after.loc[10, "hi"]), "n": int(full["n"].iloc[0])}
        for ax, cc in ((axes[0], full), (axes[1], after)):
            ax.plot(cc.index, cc["car"] * 1e4, color=c, lw=2, label=f"{g} (n={int(cc['n'].iloc[0]):,})")
            ax.fill_between(cc.index, cc["lo"] * 1e4, cc["hi"] * 1e4, color=c, alpha=0.15, lw=0)
    axes[0].axvspan(-0.5, 1.5, color="#f2efe6", zorder=0)
    axes[0].text(0.5, axes[0].get_ylim()[1] * 0.92, "publication\ntime unknown", ha="center", va="top", color=GREY, fontsize=8)
    axes[0].set_title("Cumulative abnormal return around the news", loc="left")
    axes[1].set_title("From day 2, when you could actually trade", loc="left")
    for ax in axes:
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_xlabel("trading days relative to the news")
    axes[0].set_ylabel("basis points vs QQQ")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "event_study.png", dpi=160)

    # long-short: honest timing against the version that peeks at day 0 and day 1
    fig, ax = plt.subplots(figsize=(7, 4))
    res["long_short"] = {}
    for lag, label, c, ls in [(0, "enter before day 0 (impossible, peeks)", GREY, ":"),
                              (1, "enter before day 1 (still peeks if news came after the close)", BLUE, "--"),
                              (2, "enter at close of day 1 (tradable)", ORANGE, "-")]:
        pnl = events.news_long_short(sig, ar, entry_lag=lag, hold=5, threshold=THRESH)
        res["long_short"][f"entry_lag_{lag}"] = {"sharpe": sharpe(pnl), "ann_return": float(pnl.mean() * 252)}
        ax.plot((1 + pnl).cumprod(), color=c, ls=ls, lw=1.8, label=f"{label}, Sharpe {sharpe(pnl):.2f}")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("growth of 1, before costs (log)")
    ax.set_title("Long positive news, short negative news, hold 5 days", loc="left")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "long_short.png", dpi=160)

    (ROOT / "results" / "event_study.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
