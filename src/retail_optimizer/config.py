from pathlib import Path

_cwd = Path.cwd()
ROOT_DIR = _cwd if (_cwd / "app.py").exists() else Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

RANDOM_SEED = 42

CATEGORIES = [
    "Fresh Produce",
    "Dairy",
    "Bakery",
    "Beverages",
    "Snacks",
    "Household",
    "Personal Care",
    "Frozen",
]

SHELF_LEVELS = ["bottom", "waist", "eye", "top", "endcap"]
