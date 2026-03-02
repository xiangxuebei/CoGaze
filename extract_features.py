from __future__ import annotations

import argparse
from pathlib import Path

from cogaze.features.extractors import export_feature_records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Extract oculomotor features from predicted gaze")
    p.add_argument("--gaze-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/features"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path, detail_path = export_feature_records(args.gaze_root, args.output)
    print(f"Saved feature table to {csv_path}")
    print(f"Saved detailed metadata to {detail_path}")


if __name__ == "__main__":
    main()
