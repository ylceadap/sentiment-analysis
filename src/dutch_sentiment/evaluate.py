"""Print stored held-out evaluation evidence without reusing the test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    """Print stored held-out metrics without re-evaluating the model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="artifacts/final_metrics.json")
    args = parser.parse_args()
    path = Path(args.metrics)
    if not path.is_file():
        raise FileNotFoundError(f"Metrics artifact not found; run training first: {path}")
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
