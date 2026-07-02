"""Tests for the two-phase next-period model."""
from cycle_predictor.data import Cycle
from cycle_predictor.models.twophase import TwoPhaseModel, luteal_lengths


def mk(length, ov, confirmed=True):
    return Cycle(user_id="u", cycle_number=1, cycle_length_days=length,
                 estimated_ovulation_day=ov, extra={"ovulation_confirmed": confirmed})


def test_luteal_lengths():
    assert luteal_lengths([mk(28, 14), mk(30, 16), mk(26, 12)]) == [14, 14, 14]


def test_fit_and_predict_from_ovulation():
    m = TwoPhaseModel.fit([mk(28, 14), mk(30, 15), mk(29, 15)])  # luteal 14,15,14
    assert 13.5 < m.luteal_mean < 15
    pred, sd = m.predict_from_ovulation(16)
    assert abs(pred - (16 + m.luteal_mean)) < 1e-9
    assert sd >= 0


def test_confirmed_only_filter():
    cs = [mk(28, 14, confirmed=True), mk(40, 10, confirmed=False)]
    assert luteal_lengths(cs, confirmed_only=True) == [14]
    assert len(luteal_lengths(cs, confirmed_only=False)) == 2
