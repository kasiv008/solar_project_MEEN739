#!/bin/bash
# Full pipeline, both LiDAR vintages (~10 min on a laptop). Needs the data from solar_data.zip.
#   ./run_all.sh            everything
#   ./run_all.sh 2025       one vintage (plus the shared steps)
set -euo pipefail
cd "$(dirname "$0")"
R=./run.sh
VINTAGES="${1:-2017 2025}"

step() { echo; echo "=== $*"; }

if [ ! -f data/raw/tamu_fcor.gpkg ]; then
  step "01 download TAMU GIS layers (needs internet)"; $R scripts/01_fetch_tamu_layers.py
fi
if [ ! -f tamu_land_resources/laz/merged_tamu_2025.laz ] && ls data/raw/lidar_2025/tiles/*.laz >/dev/null 2>&1; then
  step "merge 2025 StratMap tiles"; $R scripts/merge_lidar_2025.py
fi

step "02 screen footprints + open land"; $R scripts/02_screen_footprints.py
$R scripts/plot_step1_map.py

for v in $VINTAGES; do
  step "[$v] 03 LiDAR rasters";        $R scripts/03_lidar_rasters.py --vintage "$v"
  step "[$v] 04 roof geometry";        $R scripts/04_roof_geometry.py --vintage "$v"
  step "[$v] 05 shading + SAM inputs"; $R scripts/05_shading.py --vintage "$v"
  step "[$v] 07 parking carports";     $R scripts/07_parking.py --vintage "$v"
  step "[$v] 06 QA vs LiDAR/imagery";  $R scripts/06_qa_imagery.py --vintage "$v"
  step "[$v] 08 PySAM per site";       $R scripts/08_pysam_batch.py --vintage "$v"
  step "[$v] 09 campus balance";       $R scripts/09_campus_balance.py --vintage "$v"
done

if [[ "$VINTAGES" == *2017* && "$VINTAGES" == *2025* ]]; then
  step "compare 2017 vs 2025"; $R scripts/compare_vintages.py
fi

step "10 design sweeps"; $R scripts/10_design_sweeps.py; $R scripts/plot_design_sweeps.py

for v in $VINTAGES; do
  step "[$v] 11 portfolios"; $R scripts/11_portfolios.py --vintage "$v"; $R scripts/plot_portfolios.py --vintage "$v"
done
step "12 web app data"; $R scripts/12_build_webapp.py
echo; echo "done - see outputs/, and ./serve.sh -> http://localhost:8000"
