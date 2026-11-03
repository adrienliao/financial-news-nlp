import numpy as np
import pandas as pd
import pytest

from newsalpha import events

DAYS = pd.bdate_range("2021-01-04", periods=40)


def test_weekend_news_maps_to_monday():
    dates = pd.Series(pd.to_datetime(["2021-01-09", "2021-01-11", "2021-01-08"]))
    out = events.align_to_trading_days(dates, DAYS)
    assert list(out.dt.strftime("%a")) == ["Mon", "Mon", "Fri"]


def test_news_after_last_day_is_dropped():
    dates = pd.Series(pd.to_datetime(["2030-01-01"]))
    assert events.align_to_trading_days(dates, DAYS).isna().all()


def test_event_window_picks_the_right_returns():
    ar = pd.DataFrame({"A": np.arange(40, dtype=float)}, index=DAYS)
    ev = pd.DataFrame({"day0": [DAYS[10]], "ticker": ["A"]})
    w = events.event_window(ev, ar, lo=-2, hi=3)
    assert w.iloc[0].tolist() == [8, 9, 10, 11, 12, 13]


def test_event_window_too_close_to_edges_is_nan():
    ar = pd.DataFrame({"A": np.zeros(40)}, index=DAYS)
    ev = pd.DataFrame({"day0": [DAYS[1]], "ticker": ["A"]})
    assert events.event_window(ev, ar, lo=-5, hi=3).isna().all().all()


def test_long_short_only_trades_after_day_one():
    """A huge move on day 0 and day 1 must not be captured, a move on day 2 must."""
    ar = pd.DataFrame(0.0, index=DAYS, columns=["A", "B"])
    d0 = 10
    ar.iloc[d0, 0] = ar.iloc[d0 + 1, 0] = 0.5
    ar.iloc[d0 + 2, 0] = 0.01
    sig = pd.DataFrame({"day0": [DAYS[d0]], "ticker": ["A"], "score": [0.9]})
    pnl = events.news_long_short(sig, ar, entry_lag=2, hold=1)
    assert pnl.sum() == pytest.approx(0.01)
    assert pnl.iloc[d0 + 2] == pytest.approx(0.01)


def test_short_side_and_threshold():
    ar = pd.DataFrame(0.0, index=DAYS, columns=["A", "B"])
    ar.iloc[12, 1] = -0.02
    sig = pd.DataFrame({"day0": [DAYS[10], DAYS[10]], "ticker": ["B", "A"], "score": [-0.8, 0.1]})
    pnl = events.news_long_short(sig, ar, entry_lag=2, hold=1, threshold=0.5)
    assert pnl.sum() == pytest.approx(0.02)   # short B earns, A is below threshold


def test_car_confidence_band_contains_mean():
    w = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (500, 6)), columns=range(-2, 4))
    c = events.car_with_ci(w, start=0)
    assert (c["lo"] <= c["car"]).all() and (c["car"] <= c["hi"]).all()
    assert list(c.index) == [0, 1, 2, 3]
