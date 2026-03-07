"""
A/B Testing — traffic splitting, Welch's t-test, statistical significance.

All statistics are implemented from scratch using the ``math`` standard-
library module only.  No scipy/statsmodels dependency required.

Statistical approach
--------------------
* Welch's t-test (unequal variances) for the two-sample comparison.
* Two-tailed p-value approximated via a rational polynomial approximation
  of the Student's t CDF (Abramowitz & Stegun 26.7.8).
* Default significance threshold: α = 0.05.
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Tuple


class ABTest:
    """
    Online A/B test manager.

    Parameters
    ----------
    name : str
        Human-readable test identifier.
    control_model : str
        Name / ID of the control (baseline) model.
    treatment_model : str
        Name / ID of the treatment (challenger) model.
    traffic_split : float
        Fraction of traffic routed to the **treatment** arm.
        Must be in (0, 1).  Default: 0.5 (50/50 split).
    """

    def __init__(
        self,
        name: str,
        control_model: str,
        treatment_model: str,
        traffic_split: float = 0.5,
    ) -> None:
        if not 0 < traffic_split < 1:
            raise ValueError("traffic_split must be strictly between 0 and 1")
        self.name = name
        self.control_model = control_model
        self.treatment_model = treatment_model
        self.traffic_split = traffic_split

        self._control_outcomes: List[float] = []
        self._treatment_outcomes: List[float] = []

    # ------------------------------------------------------------------
    # Data recording
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        variant: Literal["control", "treatment"],
        metric_value: float,
    ) -> None:
        """
        Record a single outcome observation for *variant*.

        Parameters
        ----------
        variant : "control" | "treatment"
            Which arm produced this observation.
        metric_value : float
            Observed metric (e.g. conversion probability, latency ms,
            revenue, accuracy score).
        """
        if variant == "control":
            self._control_outcomes.append(float(metric_value))
        elif variant == "treatment":
            self._treatment_outcomes.append(float(metric_value))
        else:
            raise ValueError(f"variant must be 'control' or 'treatment', got {variant!r}")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, object]:
        """
        Compute summary statistics for the current test state.

        Returns
        -------
        dict with keys:
            * ``control_mean``   – mean of control outcomes
            * ``treatment_mean`` – mean of treatment outcomes
            * ``lift_pct``       – (treatment_mean - control_mean) / |control_mean| × 100
            * ``p_value``        – Welch t-test p-value (two-tailed)
            * ``significant``    – bool, True if p_value < 0.05
            * ``n_control``      – number of control observations
            * ``n_treatment``    – number of treatment observations
            * ``t_stat``         – Welch t-statistic

        Raises
        ------
        ValueError
            If either arm has fewer than 2 observations.
        """
        a = self._control_outcomes
        b = self._treatment_outcomes
        if len(a) < 2 or len(b) < 2:
            raise ValueError(
                "Need ≥2 observations per arm to compute statistics. "
                f"Got control={len(a)}, treatment={len(b)}"
            )

        control_mean = _mean(a)
        treatment_mean = _mean(b)

        denom = abs(control_mean) if control_mean != 0 else 1.0
        lift_pct = (treatment_mean - control_mean) / denom * 100.0

        t_stat, p_value = self._welch_t_test(a, b)

        return {
            "control_mean": control_mean,
            "treatment_mean": treatment_mean,
            "lift_pct": lift_pct,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "n_control": len(a),
            "n_treatment": len(b),
            "t_stat": t_stat,
        }

    def _welch_t_test(
        self,
        a: List[float],
        b: List[float],
    ) -> Tuple[float, float]:
        """
        Two-sample Welch's t-test (does **not** assume equal variances).

        Parameters
        ----------
        a, b : list[float]
            Sample observations for each arm.

        Returns
        -------
        (t_stat, p_value) : (float, float)
            Two-tailed Welch t-statistic and corresponding p-value.

        Notes
        -----
        Degrees of freedom via the Welch–Satterthwaite equation.
        p-value uses a rational polynomial approximation of the
        regularised incomplete beta function (Student's t CDF).
        """
        n_a, n_b = len(a), len(b)
        mean_a, mean_b = _mean(a), _mean(b)
        var_a, var_b = _variance(a), _variance(b)

        se = math.sqrt(var_a / n_a + var_b / n_b)
        if se == 0:
            return 0.0, 1.0

        t_stat = (mean_a - mean_b) / se

        # Welch–Satterthwaite degrees of freedom
        term_a = (var_a / n_a) ** 2 / (n_a - 1)
        term_b = (var_b / n_b) ** 2 / (n_b - 1)
        denom = term_a + term_b
        df = ((var_a / n_a + var_b / n_b) ** 2) / denom if denom > 0 else n_a + n_b - 2

        p_value = _t_cdf_two_tailed(t_stat, df)
        return t_stat, p_value

    def declare_winner(self, alpha: float = 0.05) -> str:
        """
        Declare the winning variant or report inconclusive.

        Parameters
        ----------
        alpha : float
            Significance level.  Default: 0.05.

        Returns
        -------
        "control" | "treatment" | "inconclusive"
        """
        stats = self.get_stats()
        if stats["p_value"] >= alpha:
            return "inconclusive"
        # Significant — pick the arm with the higher mean
        if stats["treatment_mean"] > stats["control_mean"]:
            return "treatment"
        return "control"

    @property
    def n_control(self) -> int:
        return len(self._control_outcomes)

    @property
    def n_treatment(self) -> int:
        return len(self._treatment_outcomes)


# ---------------------------------------------------------------------------
# Pure-math helpers (no numpy/scipy)
# ---------------------------------------------------------------------------

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def _variance(xs: List[float]) -> float:
    """Sample variance (Bessel-corrected, ddof=1)."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def _regularised_incomplete_beta(x: float, a: float, b: float, max_iter: int = 200) -> float:
    """
    Regularised incomplete beta function I_x(a, b) via continued fraction
    (Lentz's algorithm).  Used internally for the t-distribution CDF.
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use the symmetry relation when x > (a+1)/(a+b+2) for convergence
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularised_incomplete_beta(1.0 - x, b, a, max_iter)

    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a

    # Lentz's continued fraction
    TINY = 1e-30
    f = TINY
    C = f
    D = 0.0
    for m in range(max_iter + 1):
        for step in range(2):
            if m == 0 and step == 0:
                num = 1.0
            elif step == 0:
                num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
            else:
                num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))

            D = 1.0 + num * D
            D = TINY if abs(D) < TINY else D
            D = 1.0 / D

            C = 1.0 + num / C
            C = TINY if abs(C) < TINY else C

            f *= C * D
            if abs(C * D - 1.0) < 1e-10:
                return front * f

    return front * f


def _t_cdf_two_tailed(t: float, df: float) -> float:
    """
    Two-tailed p-value for Student's t-distribution.

    Uses the regularised incomplete beta function:
        p = I_{df/(df+t²)}(df/2, 1/2)

    where I is the regularised incomplete beta function.
    """
    t2 = t * t
    x = df / (df + t2)
    p_one_tail = 0.5 * _regularised_incomplete_beta(x, df / 2.0, 0.5)
    return 2.0 * p_one_tail
