"""Proje dizinleri. Kaynak depo içinden (pip install -e .) çalışacak şekilde tasarlandı."""
from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("CLFBENCH_HOME")
    if env:
        return Path(env).resolve()
    # src/clfbench/paths.py -> depo kökü
    return Path(__file__).resolve().parents[2]


ROOT = project_root()
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"
