"""Model metadata written next to every saved model (models/<name>/model_metadata.json)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from rainpp import __version__


def write_metadata(directory: Path, *, name: str, model_type: str, features: list[str],
                   training_period: tuple[int, int], validation_period: tuple[int, int],
                   data_sources: dict[str, str], validation_metrics: dict | None = None,
                   extra: dict | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": name,
        "model_type": model_type,
        "package_version": __version__,
        "data_kind": "real",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": features,
        "training_period": list(training_period),
        "validation_period": list(validation_period),
        "test_period_used": False,
        "data_sources": data_sources,
        "validation_metrics": validation_metrics or {},
        **(extra or {}),
    }
    path = directory / "model_metadata.json"
    path.write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    return path
