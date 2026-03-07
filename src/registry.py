"""
Model Registry — file-backed (JSON) with stage promotion and rollback.

Mirrors the conceptual model of MLflow's model registry without
requiring any external service.  Supported stages:

    Staging  →  Production  →  Archived
                    ↑
                (rollback)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

VALID_STAGES = {"Staging", "Production", "Archived", "None"}


class ModelRegistry:
    """
    File-backed model registry with versioning and stage promotion.

    Parameters
    ----------
    storage_path : str | Path
        Directory where ``registry.json`` will be stored.
    """

    _FILENAME = "registry.json"

    def __init__(self, storage_path: str | Path) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._db_path = self.storage_path / self._FILENAME
        self._db: Dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Persistence
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
    # Registration
    # ------------------------------------------------------------------

    def register_model(
        self,
        name: str,
        version: str,
        artifact_path: str,
        metrics: Optional[Dict[str, float]] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Register a new model version.

        Parameters
        ----------
        name : str
            Logical model name (e.g. ``"fraud_classifier"``).
        version : str
            Semantic version string (e.g. ``"1.2.0"``).
        artifact_path : str
            URI/path to the serialised model artifact.
        metrics : dict, optional
            Evaluation metrics at registration time.
        run_id : str, optional
            ID of the experiment run that produced this model.

        Returns
        -------
        dict
            The newly created model-version record.
        """
        self._db.setdefault(name, {"versions": {}})
        model_version: Dict[str, Any] = {
            "name": name,
            "version": version,
            "artifact_path": artifact_path,
            "metrics": metrics or {},
            "run_id": run_id,
            "stage": "None",
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._db[name]["versions"][version] = model_version
        self._save()
        return model_version

    # ------------------------------------------------------------------
    # Stage management
    # ------------------------------------------------------------------

    def promote(self, name: str, version: str, stage: str) -> Dict[str, Any]:
        """
        Set the stage for a specific model version.

        When promoting to ``"Production"``, any existing Production version
        is automatically moved to ``"Archived"`` to ensure at most one live
        Production model per name.

        Parameters
        ----------
        stage : str
            One of ``"Staging"``, ``"Production"``, ``"Archived"``.

        Returns
        -------
        dict
            Updated model-version record.

        Raises
        ------
        ValueError
            If *stage* is not a valid stage label.
        """
        if stage not in VALID_STAGES:
            raise ValueError(
                f"Invalid stage '{stage}'. Choose from {sorted(VALID_STAGES)}"
            )
        mv = self._get_version(name, version)

        # Demote current production before promoting a new one
        if stage == "Production":
            for v, rec in self._db[name]["versions"].items():
                if rec["stage"] == "Production" and v != version:
                    rec["stage"] = "Archived"
                    rec["updated_at"] = time.time()

        mv["stage"] = stage
        mv["updated_at"] = time.time()
        self._save()
        return mv

    def get_production_model(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Return the current Production model version, or ``None``.

        If multiple versions are somehow in Production (shouldn't happen
        with :meth:`promote`), the one with the latest ``updated_at``
        timestamp is returned.
        """
        versions = self._db.get(name, {}).get("versions", {})
        prod = [v for v in versions.values() if v["stage"] == "Production"]
        if not prod:
            return None
        return max(prod, key=lambda v: v["updated_at"])

    def rollback(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Roll back to the previous Production version.

        The current Production version is demoted to ``"Archived"``.
        The most-recently-archived version (by ``updated_at``) **before**
        the demotion is promoted back to ``"Production"``.

        Returns
        -------
        dict or None
            The newly promoted version, or ``None`` if no previously
            archived version exists to roll back to.
        """
        # Collect pre-existing archived versions before touching anything
        versions = self._db.get(name, {}).get("versions", {})
        previously_archived = [
            v["version"]
            for v in versions.values()
            if v["stage"] == "Archived"
        ]

        current_prod = self.get_production_model(name)
        if current_prod:
            self.promote(name, current_prod["version"], "Archived")

        # Only consider versions that were already archived before the rollback call
        candidates = [
            v for v in self._db.get(name, {}).get("versions", {}).values()
            if v["version"] in previously_archived
        ]
        if not candidates:
            return None

        prev = max(candidates, key=lambda v: v["updated_at"])
        return self.promote(name, prev["version"], "Production")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list_versions(self, name: str) -> List[Dict[str, Any]]:
        """
        Return all registered versions for *name*, sorted by version string.

        Returns
        -------
        list[dict]
            Each element is a model-version record including stage and metrics.
        """
        versions = self._db.get(name, {}).get("versions", {})
        return sorted(versions.values(), key=lambda v: v["version"])

    def get_version(self, name: str, version: str) -> Dict[str, Any]:
        """Return a specific model-version record."""
        return self._get_version(name, version)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_version(self, name: str, version: str) -> Dict[str, Any]:
        try:
            return self._db[name]["versions"][version]
        except KeyError:
            raise KeyError(f"Model '{name}' version '{version}' not found")
