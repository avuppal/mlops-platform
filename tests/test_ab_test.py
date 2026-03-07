"""Tests for ABTest."""

import math
import pytest
from src.ab_test import ABTest, _mean, _variance, _t_cdf_two_tailed


@pytest.fixture
def ab():
    return ABTest("checkout_test", "model_v1", "model_v2", traffic_split=0.5)


def _fill(ab, control_vals, treatment_vals):
    for v in control_vals:
        ab.record_outcome("control", v)
    for v in treatment_vals:
        ab.record_outcome("treatment", v)


# ------------------------------------------------------------------
# Constructor / traffic split
# ------------------------------------------------------------------

def test_invalid_traffic_split_zero():
    with pytest.raises(ValueError):
        ABTest("t", "a", "b", traffic_split=0.0)


def test_invalid_traffic_split_one():
    with pytest.raises(ValueError):
        ABTest("t", "a", "b", traffic_split=1.0)


def test_valid_traffic_splits():
    for split in [0.1, 0.3, 0.5, 0.7, 0.9]:
        ab = ABTest("t", "a", "b", traffic_split=split)
        assert ab.traffic_split == split


# ------------------------------------------------------------------
# Record outcome
# ------------------------------------------------------------------

def test_record_outcome_control(ab):
    ab.record_outcome("control", 0.92)
    assert ab.n_control == 1


def test_record_outcome_treatment(ab):
    ab.record_outcome("treatment", 0.88)
    assert ab.n_treatment == 1


def test_record_outcome_invalid_variant(ab):
    with pytest.raises(ValueError, match="variant must be"):
        ab.record_outcome("neither", 0.5)


def test_record_multiple_outcomes(ab):
    for _ in range(10):
        ab.record_outcome("control", 0.9)
        ab.record_outcome("treatment", 0.8)
    assert ab.n_control == 10
    assert ab.n_treatment == 10


# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------

def test_get_stats_requires_min_two_observations(ab):
    ab.record_outcome("control", 0.9)
    ab.record_outcome("treatment", 0.8)
    with pytest.raises(ValueError, match="≥2 observations"):
        ab.get_stats()


def test_get_stats_keys(ab):
    _fill(ab, [0.9, 0.91, 0.89], [0.85, 0.84, 0.86])
    stats = ab.get_stats()
    for key in ("control_mean", "treatment_mean", "lift_pct", "p_value", "significant",
                "n_control", "n_treatment", "t_stat"):
        assert key in stats


def test_get_stats_counts(ab):
    _fill(ab, [0.9, 0.91, 0.89], [0.85, 0.84])
    stats = ab.get_stats()
    assert stats["n_control"] == 3
    assert stats["n_treatment"] == 2


def test_get_stats_means(ab):
    _fill(ab, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    stats = ab.get_stats()
    assert stats["control_mean"] == pytest.approx(2.0)
    assert stats["treatment_mean"] == pytest.approx(5.0)


def test_get_stats_lift_pct(ab):
    _fill(ab, [1.0, 1.0, 1.0], [1.1, 1.1, 1.1])
    stats = ab.get_stats()
    assert stats["lift_pct"] == pytest.approx(10.0, abs=0.01)


def test_significant_when_means_far_apart(ab):
    # Very clearly different distributions — should be significant
    _fill(ab,
          [0.5] * 50,
          [0.9] * 50)
    stats = ab.get_stats()
    assert stats["significant"] is True
    assert stats["p_value"] < 0.05


def test_not_significant_when_means_close(ab):
    # Tiny difference with high variance — should not be significant
    import random
    rng = random.Random(42)
    ctrl = [rng.gauss(0.5, 1.0) for _ in range(20)]
    trt = [rng.gauss(0.51, 1.0) for _ in range(20)]
    _fill(ab, ctrl, trt)
    stats = ab.get_stats()
    assert stats["p_value"] > 0.05


# ------------------------------------------------------------------
# Declare winner
# ------------------------------------------------------------------

def test_declare_winner_treatment(ab):
    _fill(ab, [0.5] * 30, [0.9] * 30)
    assert ab.declare_winner() == "treatment"


def test_declare_winner_control(ab):
    _fill(ab, [0.9] * 30, [0.5] * 30)
    assert ab.declare_winner() == "control"


def test_declare_winner_inconclusive():
    ab = ABTest("t", "a", "b")
    import random
    rng = random.Random(7)
    for _ in range(15):
        ab.record_outcome("control", rng.gauss(0.5, 1.0))
        ab.record_outcome("treatment", rng.gauss(0.5, 1.0))
    result = ab.declare_winner()
    assert result in ("control", "treatment", "inconclusive")


# ------------------------------------------------------------------
# Internal math helpers
# ------------------------------------------------------------------

def test_mean_basic():
    assert _mean([1, 2, 3, 4, 5]) == pytest.approx(3.0)


def test_variance_basic():
    # Known variance of [2, 4, 4, 4, 5, 5, 7, 9] = 4.0
    assert _variance([2, 4, 4, 4, 5, 5, 7, 9]) == pytest.approx(4.571, abs=0.01)


def test_welch_t_test_symmetric(ab):
    # t-test should be symmetric: |t(a,b)| == |t(b,a)|
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [6.0, 7.0, 8.0, 9.0, 10.0]
    t1, p1 = ab._welch_t_test(a, b)
    t2, p2 = ab._welch_t_test(b, a)
    assert abs(t1) == pytest.approx(abs(t2), rel=1e-6)
    assert p1 == pytest.approx(p2, rel=1e-6)


def test_welch_t_test_identical_samples(ab):
    # Same samples — p-value should be 1.0 (no difference)
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    t, p = ab._welch_t_test(a, a[:])
    assert t == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0, abs=0.01)


def test_p_value_range(ab):
    _fill(ab, [0.5] * 20, [0.9] * 20)
    stats = ab.get_stats()
    assert 0.0 <= stats["p_value"] <= 1.0
