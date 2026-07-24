"""Load project config from config.yaml."""

from pathlib import Path

import yaml


def _find_project_root():
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / "pyproject.toml").exists():
            return d
        d = d.parent
    raise FileNotFoundError("Could not find project root (no pyproject.toml)")


PROJECT_ROOT = _find_project_root()

with open(PROJECT_ROOT / "config.yaml") as f:
    RAW = yaml.safe_load(f) or {}

SCREENSHOT_DIR = Path(RAW["screenshot_dir"]).expanduser()
TESSERACT_WORKERS = RAW["tesseract_workers"]
COMMIT_BATCH_SIZE = RAW["commit_batch_size"]
DECAY_HALF_LIFE_DAYS = RAW["decay_half_life_days"]
RESULTS_PER_PAGE = RAW["results_per_page"]
FLASK_PORT = RAW["flask_port"]
