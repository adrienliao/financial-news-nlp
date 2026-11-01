"""Line up news with prices without peeking.

The headlines only carry a calendar date, not a time. A story dated D may have
been published before the open, during the session or after the close, so:

  * day 0 is the first trading day on or after D
  * the return on day 0 and day 1 may already contain the reaction
  * a strategy can only act at the close of day 1 at the earliest, so anything
    measured from day 2 on is genuinely tradable
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sentiment_score(probs: pd.DataFrame) -> pd.Series:
    """P(bullish) minus P(bearish), in [-1, 1]."""
    return probs["bullish"] - probs["bearish"]


def align_to_trading_days(dates: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    pos = trading_days.searchsorted(pd.DatetimeIndex(dates))
    out = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    ok = pos < len(trading_days)
    out[ok] = trading_days[pos[ok]]
    return out


def daily_signal(news: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per (day 0, ticker): mean score and number of headlines."""
    df = news.assign(day0=align_to_trading_days(news["date"], trading_days)).dropna(subset=["day0"])
    return df.groupby(["day0", "ticker"]).agg(score=("score", "mean"), n=("score", "size")).reset_index()


def abnormal_returns(close: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    """Market adjusted daily returns: stock minus index."""
    r = close.pct_change(fill_method=None)
    return r.sub(market.pct_change(fill_method=None), axis=0)


def event_window(events: pd.DataFrame, ar: pd.DataFrame, lo: int = -5, hi: int = 10) -> pd.DataFrame:
    """Matrix of abnormal returns, one row per event, columns = days relative to day 0."""
    idx = {d: i for i, d in enumerate(ar.index)}
    arr = ar.to_numpy()
    col = {t: j for j, t in enumerate(ar.columns)}
    rows = np.full((len(events), hi - lo + 1), np.nan)
    for k, (d, t) in enumerate(zip(events["day0"], events["ticker"])):
        if t not in col or d not in idx:
            continue
        i0 = idx[d]
        a, b = i0 + lo, i0 + hi + 1
        if a < 0 or b > len(arr):
            continue
        rows[k] = arr[a:b, col[t]]
    return pd.DataFrame(rows, index=events.index, columns=range(lo, hi + 1))


def car_with_ci(window: pd.DataFrame, start: int) -> pd.DataFrame:
    """Cumulative abnormal return from day `start`, with a 95% confidence band across events."""
    w = window.loc[:, start:].dropna()
    car = w.cumsum(axis=1)
    m, se = car.mean(), car.std() / np.sqrt(len(car))
    return pd.DataFrame({"car": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se, "n": len(car)})


def news_long_short(signal: pd.DataFrame, ar: pd.DataFrame, entry_lag: int = 2, hold: int = 5,
                    threshold: float = 0.5) -> pd.Series:
    """Daily long-short on abnormal returns.

    A stock whose score is above `threshold` (below `-threshold`) is bought
    (sold) at the close of day `entry_lag - 1`, so it earns abnormal returns
    from day `entry_lag` to day `entry_lag + hold - 1`. With the default
    entry_lag=2 the trade happens at the close of day 1, the first moment we
    are sure the story is public.
    Each side is equally weighted across the names it holds that day.
    """
    days = ar.index
    pos = pd.DataFrame(0.0, index=days, columns=ar.columns)
    idx = {d: i for i, d in enumerate(days)}
    for d, t, s in signal[["day0", "ticker", "score"]].itertuples(index=False):
        if t not in pos.columns or d not in idx or abs(s) < threshold:
            continue
        i = idx[d] + entry_lag - 1
        pos.iloc[i:i + hold, pos.columns.get_loc(t)] += np.sign(s)
    longs, shorts = pos.clip(lower=0), (-pos).clip(lower=0)
    lw = longs.div(longs.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    sw = shorts.div(shorts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    # positions decided at the close of day i earn the return of day i + 1
    return ((lw - sw).shift(1) * ar.fillna(0)).sum(axis=1)
