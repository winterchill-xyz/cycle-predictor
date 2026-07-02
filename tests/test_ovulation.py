"""Tests for the wearable thermal-shift ovulation detector."""
from cycle_predictor.models.ovulation import detect_thermal_shift


def test_detects_clear_biphasic_shift():
    temp = {d: 36.40 for d in range(0, 14)}
    temp.update({d: 36.75 for d in range(14, 24)})   # sustained rise from day 14
    est = detect_thermal_shift(temp)
    assert est is not None and 12 <= est <= 14         # ovulation ~ day before shift


def test_none_on_flat_series():
    assert detect_thermal_shift({d: 36.5 for d in range(0, 25)}) is None


def test_none_on_too_sparse():
    assert detect_thermal_shift({0: 36.5, 1: 36.6}) is None


def test_ignores_shift_outside_search_window():
    # a rise only at the very end (day 24) is outside the default search=(5,22)
    temp = {d: 36.4 for d in range(0, 24)}
    temp.update({d: 36.9 for d in range(24, 30)})
    assert detect_thermal_shift(temp) is None
