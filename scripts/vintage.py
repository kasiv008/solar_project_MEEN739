"""LiDAR vintage selection shared by the LiDAR-dependent steps (03-06).

Every step takes `--vintage 2017|2025` (default 2017) and reads/writes only its own folders:
  data/interim/lidar_<v>/   rasters          data/interim/<v>/   intermediate inventory
  outputs/<v>/              final inventory, QA
"""
import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VINTAGES = json.loads((ROOT / "config/lidar_vintages.json").read_text())


def get_vintage() -> dict:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--vintage", default="2017", choices=sorted(VINTAGES))
    args, _ = p.parse_known_args()
    v = dict(VINTAGES[args.vintage], name=args.vintage)
    v["laz_files"] = sorted(f for pat in v["laz"] for f in glob.glob(str(ROOT / pat)))
    v["lidar_dir"] = ROOT / f"data/interim/lidar_{args.vintage}"
    v["interim_dir"] = ROOT / f"data/interim/{args.vintage}"
    v["out_dir"] = ROOT / f"outputs/{args.vintage}"
    for k in ("lidar_dir", "interim_dir", "out_dir"):
        v[k].mkdir(parents=True, exist_ok=True)
    return v
