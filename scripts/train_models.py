import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retail_optimizer.modeling import train_sales_model


if __name__ == "__main__":
    metrics = train_sales_model()
    for key, value in metrics.items():
        print(f"{key}: {value}")
