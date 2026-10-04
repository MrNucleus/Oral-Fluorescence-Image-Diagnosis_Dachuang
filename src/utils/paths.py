"""项目路径解析：统一管理各目录位置，避免硬编码相对路径。"""

from pathlib import Path

# src/utils/paths.py -> parents[2] 即项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_IMAGES = DATA_RAW / "images"
DATA_ANNOTATIONS = DATA_RAW / "annotations"
DATASETS_DIR = PROJECT_ROOT / "datasets"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REPORTS_DIR = PROJECT_ROOT / "reports"
RUNS_DIR = PROJECT_ROOT / "runs"


def project_path(*parts) -> Path:
    """返回项目根目录下的路径。"""
    return PROJECT_ROOT.joinpath(*parts)


def ensure_dir(path) -> Path:
    """确保目录存在并返回 Path。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
