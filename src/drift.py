"""
Drift Detection — Population Stability Index (PSI) and Kolmogorov-Smirnov test.

PSI interpretation
------------------
* PSI < 0.10  : No significant drift
* 0.10 ≤ PSI < 0.25 : Moderate drift — investigate
* PSI ≥ 0.25  : Significant drift — retrain recommended

KS interpretation
-----------------
The two-sample KS test checks whether reference and current distributions
share the same underlying CDF.  A p-value below the threshold (default 0.05)
indicates significant drift.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Try to use scipy for the KS test p-value if available;
# fall back to a pure-Python implementation.
try:
    from scipy import stats as _scipy_stats  # type: ignore

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

_EPS = 1e-10  # numerical stability guard for log(0)


# ---------------------------------------------------------------------------
# Population Stability Index
# ---------------------------------------------------------------------------

def population_stability_index(
    expected: Sequence[float],
    actual: Sequence[float],
    bins: int = 10,
) -> float:
    """
    Compute the Population Stability Index between *expected* and *actual*.

    PSI = Σ (actual_pct_i − expected_pct_i) × ln(actual_pct_i / expected_pct_i)

    Bin edges are derived from the *expected* distribution so that results
    are comparable across time windows.

    Parameters
    ----------
    expected : sequence of float
        Reference (training-time) distribution values.
    actual : sequence of float
        Current (serving-time) distribution values.
    bins : int
        Number of equal-width bins.  Default: 10.

    Returns
    -------
    float
        PSI score.

    Raises
    ------
    ValueError
        If either sequence is empty or ``bins < 1``.
    """
    if len(expected) == 0 or len(actual) == 0:
        raise ValueError("expected and actual must be non-empty sequences")
    if bins < 1:
        raise ValueError("bins must be ≥ 1")

    exp_list = list(expected)
    act_list = list(actual)

    min_val = min(min(exp_list), min(act_list))
    max_val = max(max(exp_list), max(act_list))

    if min_val == max_val:
        # Degenerate case — single unique value, no drift possible
        return 0.0

    bin_width = (max_val - min_val) / bins
    edges = [min_val + i * bin_width for i in range(bins + 1)]
    edges[-1] = max_val + _EPS  # ensure all values fall inside

    def _bin_counts(data: List[float]) -> List[int]:
        counts = [0] * bins
        for val in data:
            idx = int((val - min_val) / bin_width)
            idx = min(idx, bins - 1)
            counts[idx] += 1
        return counts

    exp_counts = _bin_counts(exp_list)
    act_counts = _bin_counts(act_list)

    n_exp = len(exp_list)
    n_act = len(act_list)

    psi = 0.0
    for e_cnt, a_cnt in zip(exp_counts, act_counts):
        e_pct = max(e_cnt / n_exp, _EPS)
        a_pct = max(a_cnt / n_act, _EPS)
        psi += (a_pct - e_pct) * math.log(a_pct / e_pct)

    return psi


def psi_severity(psi: float) -> str:
    """Return a human-readable severity label for a PSI score."""
    if psi < 0.10:
        return "no_drift"
    if psi < 0.25:
        return "moderate_drift"
    return "significant_drift"


# ---------------------------------------------------------------------------
# Two-sample Kolmogorov-Smirnov test
# ---------------------------------------------------------------------------

def ks_drift_score(
    reference: Sequence[float],
    current: Sequence[float],
) -> Tuple[float, float]:
    """
    Two-sample Kolmogorov-Smirnov test.

    Uses ``scipy.stats.ks_2samp`` when scipy is available; otherwise falls
    back to a pure-Python implementation.

    Parameters
    ----------
    reference : sequence of float
        Reference distribution samples.
    current : sequence of float
        Current distribution samples.

    Returns
    -------
    (ks_stat, p_value) : (float, float)
        KS statistic (maximum absolute difference between empirical CDFs)
        and the asymptotic p-value.
    """
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("reference and current must be non-empty")

    if _SCIPY_AVAILABLE:
        result = _scipy_stats.ks_2samp(list(reference), list(current))
        return float(result.statistic), float(result.pvalue)

    return _ks_2samp_pure(list(reference), list(current))


def _ks_2samp_pure(
    a: List[float],
    b: List[float],
) -> Tuple[float, float]:
    """
    Pure-Python two-sample KS test.

    Algorithm
    ---------
    1. Merge and sort both samples.
    2. Walk through sorted values computing the empirical CDF for each sample.
    3. KS statistic = max |F_a(x) - F_b(x)|.
    4. p-value via the Kolmogorov distribution asymptotic approximation.
    """
    n_a = len(a)
    n_b = len(b)

    a_sorted = sorted(a)
    b_sorted = sorted(b)

    # Build all unique breakpoints
    all_vals = sorted(set(a_sorted + b_sorted))

    def _ecdf(data: List[float], val: float) -> float:
        """Empirical CDF P(X ≤ val) using binary search."""
        lo, hi = 0, len(data)
        while lo < hi:
            mid = (lo + hi) // 2
            if data[mid] <= val:
                lo = mid + 1
            else:
                hi = mid
        return lo / len(data)

    ks_stat = 0.0
    for val in all_vals:
        diff = abs(_ecdf(a_sorted, val) - _ecdf(b_sorted, val))
        if diff > ks_stat:
            ks_stat = diff

    # Asymptotic p-value via Kolmogorov distribution
    n = n_a * n_b / (n_a + n_b)
    lambda_val = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * ks_stat
    p_value = _kolmogorov_smirnov_pvalue(lambda_val)

    return ks_stat, p_value


def _kolmogorov_smirnov_pvalue(lam: float) -> float:
    """
    Kolmogorov distribution survival function P(K > lam) approximation.

    Uses the standard series: 2 Σ_{k=1}^{∞} (-1)^{k-1} exp(-2 k² λ²).
    Terminates when the incremental term is below machine epsilon.
    """
    if lam <= 0:
        return 1.0
    p = 0.0
    for k in range(1, 1000):
        term = ((-1) ** (k - 1)) * math.exp(-2 * k * k * lam * lam)
        p += term
        if abs(term) < 1e-12:
            break
    return min(max(2.0 * p, 0.0), 1.0)


# ---------------------------------------------------------------------------
# DriftMonitor
# ---------------------------------------------------------------------------

class DriftMonitor:
    """
    Multi-feature drift monitor that combines PSI and KS tests.

    Parameters
    ----------
    reference_data : dict[str, list[float]]
        Feature-name → list of reference observations.
    psi_threshold : float
        PSI value above which a feature is flagged as drifted.  Default: 0.25.
    ks_threshold : float
        KS p-value below which a feature is flagged as drifted.  Default: 0.05.
    bins : int
        Number of bins used in PSI calculation.  Default: 10.
    """

    def __init__(
        self,
        reference_data: Dict[str, List[float]],
        psi_threshold: float = 0.25,
        ks_threshold: float = 0.05,
        bins: int = 10,
    ) -> None:
        if not reference_data:
            raise ValueError("reference_data must contain at least one feature")
        self.reference_data = reference_data
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.bins = bins

    def check(
        self,
        current_data: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """
        Check each feature for distribution drift.

        Parameters
        ----------
        current_data : dict[str, list[float]]
            Feature-name → list of current observations.
            Features not present in *reference_data* are ignored.
            Features in *reference_data* missing from *current_data*
            raise ``KeyError``.

        Returns
        -------
        dict
            ``{
                "feature_name": {
                    "psi": float,
                    "ks_stat": float,
                    "ks_pvalue": float,
                    "drifted": bool,
                    "severity": str,
                },
                ...,
                "overall_drifted": bool,
            }``
        """
        result: Dict[str, Any] = {}
        any_drifted = False

        for feature, ref_vals in self.reference_data.items():
            if feature not in current_data:
                raise KeyError(
                    f"Feature '{feature}' present in reference_data but missing "
                    "from current_data"
                )
            cur_vals = current_data[feature]

            psi = population_stability_index(ref_vals, cur_vals, bins=self.bins)
            ks_stat, ks_pvalue = ks_drift_score(ref_vals, cur_vals)

            psi_drift = psi >= self.psi_threshold
            ks_drift = ks_pvalue < self.ks_threshold
            drifted = psi_drift or ks_drift

            if drifted:
                any_drifted = True

            result[feature] = {
                "psi": psi,
                "ks_stat": ks_stat,
                "ks_pvalue": ks_pvalue,
                "drifted": drifted,
                "severity": psi_severity(psi),
            }

        result["overall_drifted"] = any_drifted
        return result
