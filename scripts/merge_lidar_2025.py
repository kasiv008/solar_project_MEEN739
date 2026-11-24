"""Merge the StratMap 2025 tiles that intersect the campus tile into one LAZ, cropped to the tile.

Output: tamu_land_resources/laz/merged_tamu_2025.laz (same extent as merged_tamu.laz, 2017).
All dimensions and the classification are kept; CRS = NAD83(2011) / UTM 14N + NAVD88 (GEOID18).
"""
import glob
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TILES = sorted(glob.glob(str(ROOT / "data/raw/lidar_2025/tiles/*.laz")))
OUT = ROOT / "tamu_land_resources/laz/merged_tamu_2025.laz"
BOUNDS = "([753000, 756000], [3388500, 3391500])"

pipeline = [*TILES,
            {"type": "filters.merge"},
            {"type": "filters.crop", "bounds": BOUNDS},
            {"type": "writers.las", "filename": str(OUT), "compression": "laszip", "minor_version": 4,
             "dataformat_id": 6, "forward": "all", "extra_dims": "all"}]
subprocess.run(["pdal", "pipeline", "--stdin"], input=json.dumps(pipeline), text=True, check=True)
print(OUT, f"{OUT.stat().st_size / 1e6:.0f} MB")
