"""Cross-platform version of run_all.sh (works on Windows, macOS and Linux).

Run it with the Python of the `solar` env:
    conda activate solar
    python run_all.py              # everything, both LiDAR vintages
    python run_all.py 2025         # one vintage (plus the shared steps)
    python run_all.py --dry-run    # print the steps without running them

Each step runs in a child process with the same interpreter. The env's PROJ/GDAL data folders and,
on Windows, its DLL folders are put on the child environment, so this also works when the env
is not activated (e.g. C:\\...\\envs\\solar\\python.exe run_all.py).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WIN = os.name == "nt"


def child_env() -> dict:
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    prefix = Path(sys.prefix)
    share = prefix / "Library" / "share" if WIN else prefix / "share"
    for var, sub in (("PROJ_DATA", "proj"), ("PROJ_LIB", "proj"), ("GDAL_DATA", "gdal")):
        if (share / sub).is_dir():
            env[var] = str(share / sub)
    if WIN:  # what `conda activate` does: DLLs and the pdal CLI live under Library\bin
        dirs = [prefix, prefix / "Library" / "mingw-w64" / "bin", prefix / "Library" / "usr" / "bin",
                prefix / "Library" / "bin", prefix / "Scripts"]
        env["PATH"] = os.pathsep.join([str(d) for d in dirs if d.is_dir()] + [env.get("PATH", "")])
    return env


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vintages", nargs="*", default=["2017", "2025"], choices=["2017", "2025"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    env, t0 = child_env(), time.time()

    def run(script: str, *args: str) -> None:
        cmd = [sys.executable, "-W", "ignore", str(ROOT / "scripts" / script), *args]
        print(f"  $ python scripts/{script} {' '.join(args)}".rstrip(), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)

    def step(msg: str) -> None:
        print(f"\n=== {msg}", flush=True)

    if not (ROOT / "data/raw/tamu_fcor.gpkg").exists():
        step("01 download TAMU GIS layers (needs internet)"); run("01_fetch_tamu_layers.py")
    if (not (ROOT / "tamu_land_resources/laz/merged_tamu_2025.laz").exists()
            and any((ROOT / "data/raw/lidar_2025/tiles").glob("*.laz"))):
        step("merge 2025 StratMap tiles"); run("merge_lidar_2025.py")

    step("02 screen footprints + open land"); run("02_screen_footprints.py"); run("plot_step1_map.py")

    for v in a.vintages:
        vv = ("--vintage", v)
        step(f"[{v}] 03 LiDAR rasters");        run("03_lidar_rasters.py", *vv)
        step(f"[{v}] 04 roof geometry");        run("04_roof_geometry.py", *vv)
        step(f"[{v}] 05 shading + SAM inputs"); run("05_shading.py", *vv)
        step(f"[{v}] 07 parking carports");     run("07_parking.py", *vv)
        step(f"[{v}] 06 QA vs LiDAR/imagery");  run("06_qa_imagery.py", *vv)
        step(f"[{v}] 08 PySAM per site");       run("08_pysam_batch.py", *vv)
        step(f"[{v}] 09 campus balance");       run("09_campus_balance.py", *vv)

    if {"2017", "2025"} <= set(a.vintages):
        step("compare 2017 vs 2025"); run("compare_vintages.py")

    step("10 design sweeps"); run("10_design_sweeps.py"); run("plot_design_sweeps.py")

    for v in a.vintages:
        step(f"[{v}] 11 portfolios")
        run("11_portfolios.py", "--vintage", v); run("plot_portfolios.py", "--vintage", v)
    step("12 web app data"); run("12_build_webapp.py")
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min - see outputs/, then: "
          "python -m http.server 8000 --bind 127.0.0.1 -d webapp  ->  http://localhost:8000")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"\nstep failed (exit {e.returncode}): {' '.join(map(str, e.cmd))}")
