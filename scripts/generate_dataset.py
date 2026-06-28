import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retail_optimizer.eda import run_eda
from retail_optimizer.real_data import import_online_retail_dataset


if __name__ == "__main__":
    paths = import_online_retail_dataset()
    summary = run_eda()
    for name, path in paths.items():
        print(f"{name}: {path}")
    print("eda_summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
