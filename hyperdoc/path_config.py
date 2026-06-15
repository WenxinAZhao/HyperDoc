"""Path helpers for HyperDoc."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
HYPERGRAPH_DIR = PROJECT_ROOT / "hypergraph"
TMP_DIR = PROJECT_ROOT / "tmp"
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_COLBERT_PATH = os.environ.get("HYPERDOC_COLBERT_MODEL", "")


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_hypergraph_dir() -> Path:
    return HYPERGRAPH_DIR


def get_page_image_dir(dataset: str = "MMLongBench") -> Path:
    image_root = os.environ.get("HYPERDOC_IMAGE_ROOT")
    if image_root:
        return Path(image_root) / dataset
    return TMP_DIR / dataset


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
