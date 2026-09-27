"""Load project settings from config/settings.yaml with environment-variable overrides."""

from __future__ import annotations

import os
from itertools import pairwise
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"


class Paths(BaseModel):
    data_dir: Path
    model_dir: Path


class Domain(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    resolution_deg: float

    @model_validator(mode="after")
    def _check_bounds(self) -> Domain:
        if not (self.lat_min < self.lat_max and self.lon_min < self.lon_max):
            raise ValueError("domain min bounds must be below max bounds")
        return self


class Season(BaseModel):
    months: list[int]


class Forecast(BaseModel):
    source: str
    init_hour_utc: int
    lead_days: list[int]
    members: list[str]
    rain_day_end_hour_utc: int


class Observation(BaseModel):
    source: str


class Split(BaseModel):
    train: tuple[int, int]
    validation: tuple[int, int]
    test: tuple[int, int]

    @model_validator(mode="after")
    def _check_chronological(self) -> Split:
        periods = [self.train, self.validation, self.test]
        for start, end in periods:
            if start > end:
                raise ValueError(f"split period {start}-{end} is reversed")
        for earlier, later in pairwise(periods):
            if earlier[1] >= later[0]:
                raise ValueError("train/validation/test periods must be chronological and disjoint")
        return self


class Api(BaseModel):
    host: str
    port: int
    cors_origins: list[str]


class Settings(BaseModel):
    mode: Literal["demo", "real"]
    paths: Paths
    domain: Domain
    season: Season
    forecast: Forecast
    observation: Observation
    split: Split
    api: Api


_ENV_OVERRIDES = {
    "RAINPP_MODE": ("mode",),
    "RAINPP_DATA_DIR": ("paths", "data_dir"),
    "RAINPP_MODEL_DIR": ("paths", "model_dir"),
}


def load_yaml(name: str, config_dir: Path = DEFAULT_CONFIG_DIR) -> dict:
    """Read one YAML file from the config directory."""
    with open(config_dir / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_settings(config_dir: Path = DEFAULT_CONFIG_DIR) -> Settings:
    """Load settings.yaml, apply env overrides, and resolve relative paths against the project root."""
    raw = load_yaml("settings.yaml", config_dir)
    for env_var, keys in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value:
            target = raw
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value

    settings = Settings.model_validate(raw)
    root = config_dir.parent
    for field in ("data_dir", "model_dir"):
        path = getattr(settings.paths, field)
        if not path.is_absolute():
            setattr(settings.paths, field, root / path)
    return settings
