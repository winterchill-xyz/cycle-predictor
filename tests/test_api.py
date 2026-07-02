"""Tests for the predict_next_period product API."""
from datetime import date, timedelta

import pytest

from cycle_predictor.api import UserLog, predict_next_period


def test_basic_history_prediction():
    f = predict_next_period(UserLog(period_starts=["2026-05-04", "2026-06-01", "2026-06-30"]))
    assert f.mode == "history"
    assert f.n_history == 2
    assert f.earliest <= f.predicted_start <= f.latest
    assert 20 <= f.cycle_length_days <= 40
    # predicted date is onset + rounded cycle length
    assert (f.predicted_start - date(2026, 6, 30)).days == round(f.cycle_length_days)


def test_cold_start_single_period():
    f = predict_next_period(UserLog(period_starts=[date(2026, 6, 30)]))
    assert f.mode == "history" and f.n_history == 0
    assert 20 <= f.cycle_length_days <= 40          # population prior, still sensible


def test_requires_at_least_one_start():
    with pytest.raises(ValueError):
        predict_next_period(UserLog(period_starts=[]))


def test_lh_surge_switches_to_two_phase():
    onset = date(2026, 6, 30)
    lh = {onset + timedelta(days=15): 42.0}
    f = predict_next_period(UserLog(period_starts=["2026-06-02", "2026-06-30"], lh_tests=lh,
                                    today=onset + timedelta(days=15)))
    assert f.mode == "two_phase_lh"
    assert 26 <= f.cycle_length_days <= 32          # ~ ovulation(15) + luteal(~14)


def test_wearable_shift_used_when_no_lh():
    onset = date(2026, 6, 30)
    temp = {onset + timedelta(days=d): 36.40 for d in range(0, 14)}
    temp.update({onset + timedelta(days=d): 36.75 for d in range(14, 20)})
    f = predict_next_period(UserLog(period_starts=["2026-06-02", "2026-06-30"], wearable_temp=temp,
                                    today=onset + timedelta(days=19)))
    assert f.mode == "two_phase_wearable"


def test_higher_confidence_widens_interval():
    log = UserLog(period_starts=["2026-05-04", "2026-06-01", "2026-06-30"])
    w80 = (predict_next_period(log, confidence=0.8).latest
           - predict_next_period(log, confidence=0.8).earliest).days
    w95 = (predict_next_period(log, confidence=0.95).latest
           - predict_next_period(log, confidence=0.95).earliest).days
    assert w95 > w80


def test_days_until_counts_from_today():
    onset = date(2026, 6, 30)
    f = predict_next_period(UserLog(period_starts=["2026-06-02", "2026-06-30"],
                                    today=onset + timedelta(days=10)))
    assert f.days_until == (f.predicted_start - (onset + timedelta(days=10))).days


def test_accepts_date_objects_and_iso_strings():
    a = predict_next_period(UserLog(period_starts=[date(2026, 6, 2), date(2026, 6, 30)]))
    b = predict_next_period(UserLog(period_starts=["2026-06-02", "2026-06-30"]))
    assert a.predicted_start == b.predicted_start
