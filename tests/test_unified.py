"""Tests for the unified predictor's graceful degradation across available signals."""
from cycle_predictor.data import Cycle
from cycle_predictor.models.unified import UnifiedPredictor


def mk(length, ov, confirmed=True):
    return Cycle(user_id="u", cycle_number=1, cycle_length_days=length,
                 estimated_ovulation_day=ov, extra={"ovulation_confirmed": confirmed})


def make_predictor():
    sequences = [[28, 29, 30, 28], [30, 31, 29, 30]]
    cycles = [mk(28, 14), mk(30, 16), mk(29, 15)]        # luteal 14,14,14 → mean 14
    return UnifiedPredictor.fit(sequences, cycles)


def test_history_only_when_no_signals():
    p = make_predictor()
    pred = p.predict([28, 29, 30])
    assert pred.mode == "history"
    assert pred.cycle_length > 0 and pred.sd > 0


def test_new_user_no_history_no_signals_still_predicts():
    pred = make_predictor().predict([])                  # brand-new user, cold start
    assert pred.mode == "history"
    assert pred.cycle_length > 0


def test_uses_lh_surge_when_present():
    p = make_predictor()
    pred = p.predict([28, 29], lh_by_day={13: 8.0, 15: 40.0, 16: 12.0})
    assert pred.mode == "two_phase_lh"
    assert abs(pred.cycle_length - (15 + p.twophase.luteal_mean)) < 1e-9


def test_uses_wearable_when_only_temperature():
    p = make_predictor()
    temp = {d: 36.40 for d in range(0, 14)}
    temp.update({d: 36.75 for d in range(14, 24)})
    pred = p.predict([28, 29], temp_by_day=temp)
    assert pred.mode == "two_phase_wearable"


def test_prefers_lh_over_wearable_when_both_present():
    p = make_predictor()
    temp = {d: 36.40 for d in range(0, 14)}
    temp.update({d: 36.75 for d in range(14, 24)})
    pred = p.predict([28], lh_by_day={15: 40.0}, temp_by_day=temp)
    assert pred.mode == "two_phase_lh"


def test_empty_signal_dicts_fall_back_to_history():
    p = make_predictor()
    pred = p.predict([28, 29], lh_by_day={}, temp_by_day={})
    assert pred.mode == "history"
