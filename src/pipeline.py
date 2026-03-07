"""
Retraining Pipeline — automated train→eval→register→promote loop.

The pipeline orchestrates the full MLOps retraining lifecycle:

1. Check whether retraining is warranted (drift detected or metric degradation).
2. Train a new model candidate via the caller-supplied ``train_fn``.
3. Evaluate via ``eval_fn``.
4. Register the candidate in the model registry.
5. Promote to Production if it beats the current incumbent by at least
   ``threshold_pct`` percent on the primary evaluation metric.

Design philosophy
-----------------
* All business logic lives in this class; side-effects (training, eval)
  are injected as callables so the pipeline is fully testable with mocks.
* No coupling to any specific ML framework — bring your own train/eval.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from .drift import DriftMonitor
from .experiment import ExperimentTracker
from .registry import ModelRegistry


class RetrainingPipeline:
    """
    Automated retraining pipeline.

    Parameters
    ----------
    tracker : ExperimentTracker
        Used to log training runs and metrics.
    registry : ModelRegistry
        Used to register and promote trained models.
    drift_monitor : DriftMonitor
        Used to assess whether input distributions have shifted.
    train_fn : callable
        ``train_fn(train_data) → model`` — trains and returns a model object.
    eval_fn : callable
        ``eval_fn(model, eval_data) → dict[str, float]``
        — evaluates the model and returns a metrics dict.
    metric : str
        Primary metric key used to compare candidates against production.
        Default: ``"accuracy"``.
    metric_mode : str
        ``"max"`` if higher is better, ``"min"`` if lower is better.
        Default: ``"max"``.
    retrain_metric_threshold : float
        Trigger retraining when the production model's primary metric falls
        below (or above, for ``"min"`` mode) this value.
        Default: ``0.80``.
    """

    def __init__(
        self,
        tracker: ExperimentTracker,
        registry: ModelRegistry,
        drift_monitor: DriftMonitor,
        train_fn: Callable[[Any], Any],
        eval_fn: Callable[[Any, Any], Dict[str, float]],
        metric: str = "accuracy",
        metric_mode: str = "max",
        retrain_metric_threshold: float = 0.80,
    ) -> None:
        self.tracker = tracker
        self.registry = registry
        self.drift_monitor = drift_monitor
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.metric = metric
        self.metric_mode = metric_mode
        self.retrain_metric_threshold = retrain_metric_threshold

        self._run_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def should_retrain(
        self,
        current_metrics: Dict[str, float],
        drift_report: Dict[str, Any],
    ) -> bool:
        """
        Decide whether a new training run is warranted.

        Triggers retraining when **either** condition holds:

        * ``drift_report["overall_drifted"]`` is ``True``.
        * The primary metric in *current_metrics* has degraded past the
          configured threshold.

        Parameters
        ----------
        current_metrics : dict[str, float]
            Performance metrics of the currently deployed model.
        drift_report : dict
            Output of :meth:`DriftMonitor.check`.

        Returns
        -------
        bool
        """
        if drift_report.get("overall_drifted", False):
            return True

        metric_value = current_metrics.get(self.metric)
        if metric_value is None:
            return False

        if self.metric_mode == "max":
            return metric_value < self.retrain_metric_threshold
        else:  # min mode: retrain if metric exceeded threshold
            return metric_value > self.retrain_metric_threshold

    # ------------------------------------------------------------------
    # Full pipeline run
    # ------------------------------------------------------------------

    def run(
        self,
        experiment_name: str,
        train_data: Any,
        eval_data: Any,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a full train → eval → register → (conditionally promote) cycle.

        Parameters
        ----------
        experiment_name : str
            Experiment name under which the run is tracked.
        train_data : any
            Passed directly to ``train_fn``.
        eval_data : any
            Passed directly to ``eval_fn``.
        model_name : str, optional
            Registry model name.  Defaults to *experiment_name*.

        Returns
        -------
        dict
            ``{
                "run_id": str,
                "version": str,
                "metrics": dict,
                "promoted": bool,
            }``
        """
        model_name = model_name or experiment_name
        version = _generate_version()

        # 1. Start experiment run
        run_id = self.tracker.start_run(experiment_name, params={"version": version})

        try:
            # 2. Train
            model = self.train_fn(train_data)

            # 3. Evaluate
            metrics = self.eval_fn(model, eval_data)
            for key, val in metrics.items():
                self.tracker.log_metric(run_id, key, val)

            # 4. Register
            artifact_path = f"models/{model_name}/{version}/model"
            mv = self.registry.register_model(
                name=model_name,
                version=version,
                artifact_path=artifact_path,
                metrics=metrics,
                run_id=run_id,
            )
            self.registry.promote(model_name, version, "Staging")

            # 5. Conditionally promote to Production
            promoted = self.promote_if_better(model_name, version, self.metric)

            self.tracker.end_run(run_id, "FINISHED")

        except Exception as exc:
            self.tracker.end_run(run_id, "FAILED")
            raise RuntimeError(f"Pipeline run failed: {exc}") from exc

        result = {
            "run_id": run_id,
            "version": version,
            "metrics": metrics,
            "promoted": promoted,
        }
        self._run_log.append(result)
        return result

    # ------------------------------------------------------------------
    # Promotion logic
    # ------------------------------------------------------------------

    def promote_if_better(
        self,
        name: str,
        candidate_version: str,
        metric: str,
        threshold_pct: float = 0.02,
    ) -> bool:
        """
        Promote *candidate_version* to Production only if it outperforms
        the current Production model by at least *threshold_pct* percent.

        Parameters
        ----------
        name : str
            Registry model name.
        candidate_version : str
            Version string of the model to (potentially) promote.
        metric : str
            Metric key used for comparison.
        threshold_pct : float
            Minimum relative improvement required.  Default: 0.02 (2 %).

        Returns
        -------
        bool
            ``True`` if the candidate was promoted.
        """
        candidate = self.registry.get_version(name, candidate_version)
        candidate_score = candidate["metrics"].get(metric)
        if candidate_score is None:
            return False

        current_prod = self.registry.get_production_model(name)
        if current_prod is None:
            # No incumbent — always promote the first candidate
            self.registry.promote(name, candidate_version, "Production")
            return True

        prod_score = current_prod["metrics"].get(metric)
        if prod_score is None:
            self.registry.promote(name, candidate_version, "Production")
            return True

        if self.metric_mode == "max":
            required = prod_score * (1.0 + threshold_pct)
            should_promote = candidate_score >= required
        else:
            required = prod_score * (1.0 - threshold_pct)
            should_promote = candidate_score <= required

        if should_promote:
            self.registry.promote(name, candidate_version, "Production")
            return True
        return False

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def run_history(self) -> List[Dict[str, Any]]:
        """Return a copy of all pipeline run results."""
        return list(self._run_log)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _generate_version() -> str:
    """Generate a pseudo-semantic version string based on wall-clock time."""
    ts = int(time.time())
    uid = uuid.uuid4().hex[:6]
    return f"1.0.{ts % 100000}-{uid}"
