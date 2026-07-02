"""The pure-Python portable kernel must match the numpy research models exactly,
so a cross-language port validated against it is validated against the real model."""
import pytest

from cycle_predictor import portable
from cycle_predictor.api import _BACKBONE_DEFAULT, _LUTEAL_DEFAULT, default_model

pytest.importorskip("numpy")  # the reference numpy models need numpy


def test_default_params_match_api_constants():
    bp = portable.DEFAULT_PARAMS["backbone"]
    for k, v in _BACKBONE_DEFAULT.items():
        assert bp[k] == v, f"backbone param {k} drifted"
    tp = portable.DEFAULT_PARAMS["twophase"]
    assert tp["luteal_mean"] == _LUTEAL_DEFAULT["luteal_mean"]
    assert tp["luteal_sd"] == _LUTEAL_DEFAULT["luteal_sd"]


@pytest.mark.parametrize("history", [[], [28], [30, 29, 31], [27, 28, 29, 26, 30, 40]])
def test_backbone_matches_numpy(history):
    model = default_model()
    ref_mean, ref_sd = model.backbone.predict(history)
    mean, sd = portable.backbone_predict(portable.DEFAULT_PARAMS["backbone"], history)
    assert abs(mean - ref_mean) < 1e-6
    assert abs(sd - ref_sd) < 1e-6


def test_predict_matches_unified_history():
    p = portable.predict(portable.DEFAULT_PARAMS, [28, 29, 30])
    ref = default_model().predict([28, 29, 30])
    assert p["mode"] == ref.mode == "history"
    assert abs(p["cycle_length"] - ref.cycle_length) < 1e-6
    assert abs(p["sd"] - ref.sd) < 1e-6


def test_predict_matches_unified_lh_and_wearable():
    ref_model = default_model()
    lh = {15: 42.0}
    p = portable.predict(portable.DEFAULT_PARAMS, [28], lh_by_day=lh)
    ref = ref_model.predict([28], lh_by_day=lh)
    assert p["mode"] == ref.mode == "two_phase_lh"
    assert abs(p["cycle_length"] - ref.cycle_length) < 1e-9

    temp = {d: 36.40 for d in range(0, 14)}
    temp.update({d: 36.75 for d in range(14, 24)})
    p2 = portable.predict(portable.DEFAULT_PARAMS, [28], temp_by_day=temp)
    ref2 = ref_model.predict([28], temp_by_day=temp)
    assert p2["mode"] == ref2.mode == "two_phase_wearable"
    assert abs(p2["cycle_length"] - ref2.cycle_length) < 1e-9
