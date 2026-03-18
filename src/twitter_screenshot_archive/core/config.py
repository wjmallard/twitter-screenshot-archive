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


_PROJECT_ROOT = _find_project_root()

_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

with open(_CONFIG_PATH) as f:
    _raw = yaml.safe_load(f)

SCREENSHOT_DIR = Path(_raw["screenshot_dir"]).expanduser()
TESSERACT_WORKERS = _raw["tesseract_workers"]
COMMIT_BATCH_SIZE = _raw["commit_batch_size"]
DECAY_HALF_LIFE_DAYS = _raw["decay_half_life_days"]
RESULTS_PER_PAGE = _raw["results_per_page"]
FLASK_PORT = _raw["flask_port"]
