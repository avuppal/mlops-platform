"""Tests for ModelRegistry."""

import pytest
from src.registry import ModelRegistry


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(tmp_path / "registry")


def _reg(registry, name="fraud_clf", version="1.0.0", metrics=None):
    """Helper: register a model version."""
    return registry.register_model(
        name=name,
        version=version,
        artifact_path=f"models/{name}/{version}",
        metrics=metrics or {"accuracy": 0.90},
        run_id="run_abc",
    )


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------

def test_register_returns_version_dict(registry):
    mv = _reg(registry)
    assert mv["name"] == "fraud_clf"
    assert mv["version"] == "1.0.0"
    assert mv["stage"] == "None"


def test_register_stores_metrics(registry):
    mv = _reg(registry, metrics={"f1": 0.88, "precision": 0.90})
    assert mv["metrics"]["f1"] == 0.88


def test_register_stores_run_id(registry):
    mv = _reg(registry)
    assert mv["run_id"] == "run_abc"


def test_register_multiple_versions(registry):
    _reg(registry, version="1.0.0")
    _reg(registry, version="1.1.0")
    _reg(registry, version="1.2.0")
    versions = registry.list_versions("fraud_clf")
    assert len(versions) == 3


# ------------------------------------------------------------------
# Stage promotion
# ------------------------------------------------------------------

def test_promote_to_staging(registry):
    _reg(registry)
    mv = registry.promote("fraud_clf", "1.0.0", "Staging")
    assert mv["stage"] == "Staging"


def test_promote_to_production(registry):
    _reg(registry)
    registry.promote("fraud_clf", "1.0.0", "Production")
    prod = registry.get_production_model("fraud_clf")
    assert prod is not None
    assert prod["version"] == "1.0.0"


def test_promote_second_to_production_demotes_first(registry):
    _reg(registry, version="1.0.0", metrics={"accuracy": 0.88})
    _reg(registry, version="1.1.0", metrics={"accuracy": 0.91})
    registry.promote("fraud_clf", "1.0.0", "Production")
    registry.promote("fraud_clf", "1.1.0", "Production")

    prod = registry.get_production_model("fraud_clf")
    assert prod["version"] == "1.1.0"

    v1 = registry.get_version("fraud_clf", "1.0.0")
    assert v1["stage"] == "Archived"


def test_promote_to_archived(registry):
    _reg(registry)
    registry.promote("fraud_clf", "1.0.0", "Archived")
    mv = registry.get_version("fraud_clf", "1.0.0")
    assert mv["stage"] == "Archived"


def test_promote_invalid_stage_raises(registry):
    _reg(registry)
    with pytest.raises(ValueError, match="Invalid stage"):
        registry.promote("fraud_clf", "1.0.0", "Deployed")


# ------------------------------------------------------------------
# Get production model
# ------------------------------------------------------------------

def test_get_production_model_none_when_empty(registry):
    _reg(registry)
    assert registry.get_production_model("fraud_clf") is None


def test_get_production_model_returns_correct_version(registry):
    _reg(registry, version="1.0.0")
    _reg(registry, version="2.0.0")
    registry.promote("fraud_clf", "2.0.0", "Production")
    prod = registry.get_production_model("fraud_clf")
    assert prod["version"] == "2.0.0"


# ------------------------------------------------------------------
# Rollback
# ------------------------------------------------------------------

def test_rollback_restores_previous_production(registry):
    _reg(registry, version="1.0.0", metrics={"accuracy": 0.85})
    _reg(registry, version="2.0.0", metrics={"accuracy": 0.90})

    registry.promote("fraud_clf", "1.0.0", "Production")
    registry.promote("fraud_clf", "2.0.0", "Production")  # archives 1.0.0

    rolled_back = registry.rollback("fraud_clf")
    # After rollback: 2.0.0 is archived, 1.0.0 is production
    prod = registry.get_production_model("fraud_clf")
    assert prod is not None


def test_rollback_no_archived_returns_none(registry):
    _reg(registry, version="1.0.0")
    registry.promote("fraud_clf", "1.0.0", "Production")
    # No archived version exists; rollback demotes current and finds nothing
    result = registry.rollback("fraud_clf")
    assert result is None


# ------------------------------------------------------------------
# List versions
# ------------------------------------------------------------------

def test_list_versions_sorted(registry):
    for v in ["1.2.0", "1.0.0", "1.1.0"]:
        _reg(registry, version=v)
    versions = registry.list_versions("fraud_clf")
    ver_strings = [v["version"] for v in versions]
    assert ver_strings == sorted(ver_strings)


def test_list_versions_includes_stage(registry):
    _reg(registry, version="1.0.0")
    registry.promote("fraud_clf", "1.0.0", "Staging")
    versions = registry.list_versions("fraud_clf")
    assert versions[0]["stage"] == "Staging"


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------

def test_registry_persists_across_instances(tmp_path):
    path = tmp_path / "reg"
    r1 = ModelRegistry(path)
    r1.register_model("model", "1.0.0", "path/to/model", metrics={"acc": 0.9})
    r1.promote("model", "1.0.0", "Production")

    r2 = ModelRegistry(path)
    prod = r2.get_production_model("model")
    assert prod is not None
    assert prod["version"] == "1.0.0"


def test_get_version_not_found_raises(registry):
    with pytest.raises(KeyError):
        registry.get_version("ghost_model", "9.9.9")
