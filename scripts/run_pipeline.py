"""Run the full pipeline in order, from downloaded data to validation reports.

Each step is a separate script and can also be run on its own. Test years are never scored here;
the one-time test evaluation is a separate, deliberate step.

Usage:
    python scripts/run_pipeline.py                 # everything after the downloads
    python scripts/run_pipeline.py --from phase5   # resume from a step
    python scripts/run_pipeline.py --download      # also run the (long) downloads first
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
STEPS = [
    ("download", ["download_data.py"]),
    ("tables", ["build_dataset.py"]),
    ("districts", ["build_districts.py"]),
    ("phase4", ["run_baselines.py"]),
    ("regimes", ["build_regimes.py"]),
    ("phase5", ["run_regime_classifier.py"]),
    ("phase6", ["run_regime_correction.py"]),
    ("phase6b", ["run_regime_qm.py"]),
    ("phase7", ["run_heavy_rain.py"]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", choices=[name for name, _ in STEPS], default="tables")
    parser.add_argument("--download", action="store_true", help="start with the downloads")
    args = parser.parse_args()
    start = "download" if args.download else args.start
    names = [name for name, _ in STEPS]
    for name, command in STEPS[names.index(start):]:
        began = time.time()
        print(f"=== {name}: {' '.join(command)}", flush=True)
        result = subprocess.run([sys.executable, str(SCRIPTS / command[0]), *command[1:]], check=False)
        if result.returncode != 0:
            sys.exit(f"step {name} failed (exit {result.returncode})")
        print(f"=== {name} done in {time.time() - began:.0f}s", flush=True)


if __name__ == "__main__":
    main()
