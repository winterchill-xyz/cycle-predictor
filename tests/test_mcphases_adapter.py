"""Smoke tests for the mcPHASES adapter. Skipped if the data hasn't been extracted."""
import statistics

import pytest

from cycle_predictor.data import mcphases

pytestmark = pytest.mark.skipif(
    not (mcphases.DEFAULT_DIR / "hormones_and_selfreport.csv").exists(),
    reason="mcPHASES not extracted (credentialed; see data/DATASETS.md)",
)


def test_loads_cycles():
    cycles = mcphases.load()
    assert len(cycles) > 100
    assert len({c.user_id for c in cycles}) > 40


def test_cycle_lengths_plausible():
    lengths = [c.cycle_length_days for c in mcphases.load() if c.cycle_length_days]
    assert 25 < statistics.mean(lengths) < 35
    assert all(15 <= x <= 60 for x in lengths)


def test_ovulation_day_from_lh_is_physiological():
    ov = [c.estimated_ovulation_day for c in mcphases.load()
          if c.estimated_ovulation_day is not None]
    assert ov, "no LH-derived ovulation days"
    # ovulation typically ~2 weeks after menses onset
    assert 8 <= statistics.median(ov) <= 22
