from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from rainpp.config import DEFAULT_CONFIG_DIR, load_settings, load_yaml


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    target = tmp_path / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, target)
    return target


def _edit_settings(config_dir: Path, **changes: object) -> None:
    path = config_dir / "settings.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw.update(changes)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")


def test_project_config_loads() -> None:
    settings = load_settings()
    assert settings.mode in ("demo", "real")
    assert settings.paths.data_dir.is_absolute()
    assert settings.split.train[1] < settings.split.validation[0] < settings.split.test[0]


def test_env_overrides_mode_and_data_dir(config_dir: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RAINPP_MODE", "real")
    monkeypatch.setenv("RAINPP_DATA_DIR", str(tmp_path / "elsewhere"))
    settings = load_settings(config_dir)
    assert settings.mode == "real"
    assert settings.paths.data_dir == tmp_path / "elsewhere"


def test_invalid_mode_rejected(config_dir: Path, monkeypatch) -> None:
    monkeypatch.setenv("RAINPP_MODE", "production")
    with pytest.raises(ValidationError):
        load_settings(config_dir)


def test_overlapping_split_rejected(config_dir: Path) -> None:
    _edit_settings(config_dir, split={"train": [2010, 2016], "validation": [2016, 2017], "test": [2018, 2019]})
    with pytest.raises(ValidationError, match="chronological"):
        load_settings(config_dir)


def test_threshold_config_is_ordered() -> None:
    thresholds = load_yaml("thresholds.yaml")["thresholds_mm"]
    assert thresholds["heavy"] < thresholds["very_heavy"] < thresholds["extremely_heavy"]


def test_every_regime_rule_declares_evidence_level() -> None:
    regimes = load_yaml("regimes.yaml")
    rules = [regimes["synoptic"]["active_break"], regimes["synoptic"]["depression"], *regimes["local"].values()]
    assert all(rule["evidence"] in ("published", "derived", "heuristic") for rule in rules)
