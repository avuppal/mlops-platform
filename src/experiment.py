"""
Experiment Tracker — file-backed (JSON), zero external services.

Tracks runs, metrics, parameters, and enables cross-run comparison.
All data is stored as JSON on disk; no database or server required.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentTracker:
    """
    Lightweight, file-backed experiment tracker.

    Experiments map to top-level keys in a single JSON file.
    Each experiment contains a list of runs; each run stores
    params, metrics (with per-step history), start/end timestamps,
    and a terminal status.

    Parameters
    ----------
    storage_path : str | Path
        Directory where ``experiments.json`` will be read/written.
        Created automatically if it does not exist.
    """

    _FILENAME = "experiments.json"

    def __init__(self, storage_path: str | Path) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._db_path = self.storage_path / self._FILENAME
        self._db: Dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        if self._db_path.exists():
            with open(self._db_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def _save(self) -> None:
        with open(self._db_path, "w", encoding="utf-8") as fh:
            json.dump(self._db, fh, indent=2)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        experiment_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Start a new run under *experiment_name*.

        Parameters
        ----------
        experiment_name : str
            Logical grouping for a set of related runs.
        params : dict, optional
            Hyper-parameters / configuration to record at run start.

        Returns
        -------
        str
            A unique ``run_id`` (UUID4 hex string).
        """
        run_id = uuid.uuid4().hex
        run = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "params": params or {},
            "metrics": {},          # key → list of {"step": int, "value": float}
            "start_time": time.time(),
            "end_time": None,
            "status": "RUNNING",
        }
        self._db.setdefault(experiment_name, {"runs": {}})
        self._db[experiment_name]["runs"][run_id] = run
        self._save()
        return run_id

    def log_metric(
        self,
        run_id: str,
        key: str,
        value: float,
        step: int = 0,
    ) -> None:
        """
        Append a scalar metric observation for *run_id*.

        Multiple calls with the same *key* build a per-step history,
        useful for epoch-level training curves.
        """
        run = self._get_run(run_id)
        run["metrics"].setdefault(key, [])
        run["metrics"][key].append({"step": step, "value": value})
        self._save()

    def log_params(self, run_id: str, params_dict: Dict[str, Any]) -> None:
        """Merge *params_dict* into the run's recorded parameters."""
        run = self._get_run(run_id)
        run["params"].update(params_dict)
        self._save()

    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        """
        Mark a run as complete.

        Parameters
        ----------
        status : str
            Terminal status label, e.g. ``"FINISHED"``, ``"FAILED"``,
            ``"KILLED"``.
        """
        run = self._get_run(run_id)
        run["end_time"] = time.time()
        run["status"] = status
        self._save()

    # ------------------------------------------------------------------
    # Query / analysis
    # ------------------------------------------------------------------

    def get_best_run(
        self,
        experiment_name: str,
        metric: str,
        mode: str = "max",
    ) -> Dict[str, Any]:
        """
        Return the run with the best *final* value for *metric*.

        Parameters
        ----------
        metric : str
            Metric key to rank runs by.
        mode : str
            ``"max"`` (higher is better) or ``"min"`` (lower is better).

        Returns
        -------
        dict
            Full run dictionary of the winner.

        Raises
        ------
        ValueError
            If no runs exist or none recorded the requested metric.
        """
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")

        runs = self._list_runs(experiment_name)
        candidates = [
            r for r in runs if metric in r["metrics"] and r["metrics"][metric]
        ]
        if not candidates:
            raise ValueError(
                f"No runs in '{experiment_name}' recorded metric '{metric}'"
            )

        key_fn = lambda r: r["metrics"][metric][-1]["value"]  # noqa: E731
        return max(candidates, key=key_fn) if mode == "max" else min(candidates, key=key_fn)

    def compare_runs(self, run_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Side-by-side comparison of metrics for the given run IDs.

        Returns a dict keyed by *run_id*, where each value is a flat
        dict of ``{metric_key: final_value}`` plus the run ``params``.

        Example
        -------
        ::

            {
                "abc123": {"accuracy": 0.92, "loss": 0.08, "params": {...}},
                "def456": {"accuracy": 0.89, "loss": 0.11, "params": {...}},
            }
        """
        result: Dict[str, Dict[str, Any]] = {}
        for run_id in run_ids:
            run = self._get_run(run_id)
            flat_metrics = {
                k: v[-1]["value"] for k, v in run["metrics"].items() if v
            }
            result[run_id] = {**flat_metrics, "params": run["params"]}
        return result

    def get_run(self, run_id: str) -> Dict[str, Any]:
        """Return the full run dictionary for *run_id*."""
        return self._get_run(run_id)

    def list_runs(self, experiment_name: str) -> List[Dict[str, Any]]:
        """Return all runs for *experiment_name* as a list."""
        return self._list_runs(experiment_name)


    # ------------------------------------------------------------------
    # Tagging  (closes #4)
    # ------------------------------------------------------------------

    def tag_run(self, run_id: str, tags: List[str]) -> None:
        """
        Attach one or more string tags to *run_id*. Idempotent.

        Tags are stored as a list under the ``"tags"`` key in the run dict
        and persisted to the JSON backing store immediately.

        Parameters
        ----------
        run_id : str
            Target run identifier.
        tags : list[str]
            One or more tag strings to attach (e.g. ``["baseline", "v2"]``).

        Raises
        ------
        KeyError
            If *run_id* does not exist.
        """
        run = self._get_run(run_id)
        existing: List[str] = run.setdefault("tags", [])
        for tag in tags:
            if tag not in existing:
                existing.append(tag)
        self._save()

    def get_runs_by_tag(self, experiment_name: str, tag: str) -> List[Dict[str, Any]]:
        """
        Return all runs under *experiment_name* that carry *tag*.

        Parameters
        ----------
        experiment_name : str
            The experiment to search within.
        tag : str
            The tag to filter by.

        Returns
        -------
        list[dict]
            Matching run dictionaries (may be empty if no runs carry *tag*).

        Raises
        ------
        ValueError
            If *experiment_name* does not exist.
        """
        runs = self._list_runs(experiment_name)
        return [r for r in runs if tag in r.get("tags", [])]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_run(self, run_id: str) -> Dict[str, Any]:
        for exp in self._db.values():
            if run_id in exp.get("runs", {}):
                return exp["runs"][run_id]
        raise KeyError(f"run_id '{run_id}' not found")

    def _list_runs(self, experiment_name: str) -> List[Dict[str, Any]]:
        exp = self._db.get(experiment_name)
        if exp is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        return list(exp["runs"].values())
