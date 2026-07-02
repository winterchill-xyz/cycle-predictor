"""Tests for mcPHASES signal alignment. Skipped if the data isn't extracted."""
import pytest

from cycle_predictor.data import mcphases, mcphases_signals

pytestmark = pytest.mark.skipif(
    not (mcphases.DEFAULT_DIR / "computed_temperature.csv").exists(),
    reason="mcPHASES signals not extracted (credentialed; see data/DATASETS.md)",
)


def test_cycle_series_shape():
    signals = mcphases_signals.load_signals()
    cycle = mcphases.load()[0]
    series = mcphases_signals.cycle_series(cycle, signals)
    assert set(series) == {"lh", "temp", "rhr", "resp"}
    # day-of-cycle keys are non-negative and within the cycle length
    for days in series.values():
        assert all(0 <= k < cycle.cycle_length_days for k in days)


def test_signals_present_across_cycles():
    signals = mcphases_signals.load_signals()
    total = sum(len(mcphases_signals.cycle_series(c, signals)["temp"])
                for c in mcphases.load())
    assert total > 0, "no temperature aligned to any cycle"
