"""Smoke tests for the FedCycle adapter. Skipped if the CSV hasn't been downloaded."""
import pytest

from cycle_predictor.data import fedcycle

pytestmark = pytest.mark.skipif(
    not fedcycle.DEFAULT_PATH.exists(),
    reason="FedCycle CSV not downloaded (run scripts/fetch_datasets.py --only fedcycle)",
)


def test_loads_expected_shape():
    cycles = fedcycle.load()
    assert len(cycles) > 1000
    assert len({c.user_id for c in cycles}) > 100


def test_cycle_lengths_are_plausible():
    lengths = [c.cycle_length_days for c in fedcycle.load() if c.cycle_length_days]
    assert lengths, "no cycle lengths parsed"
    mean = sum(lengths) / len(lengths)
    assert 20 < mean < 40, f"implausible mean cycle length: {mean}"
    assert all(5 < x < 120 for x in lengths)


def test_canonical_fields_present():
    c = fedcycle.load()[0]
    assert c.user_id
    assert c.cycle_number >= 0
    assert "reproductive_category" in c.extra
