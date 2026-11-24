"""Overview map of step-1 screening over the 30 cm imagery (downsampled)."""
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from matplotlib.patches import Patch
from rasterio.enums import Resampling

ROOT = Path(__file__).resolve().parents[1]
TIF = ROOT / "tamu_land_resources/tif/merged_tamu_esri_30cm.tif"
G = ROOT / "data/interim/inventory_step1.gpkg"

with rasterio.open(TIF) as src:
    f = 4  # 1.2 m pixels
    img = src.read([1, 2, 3], out_shape=(3, src.height // f, src.width // f), resampling=Resampling.average)
    ext = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

roofs = gpd.read_file(G, layer="roofs_all")
open_ = gpd.read_file(G, layer="open_land_all")

fig, ax = plt.subplots(figsize=(14, 14), dpi=110)
ax.imshow(img.transpose(1, 2, 0), extent=ext)
open_[open_.is_candidate].plot(ax=ax, fc="#4caf50", ec="#1b5e20", alpha=0.45, lw=0.6)
open_[~open_.is_candidate].plot(ax=ax, fc="none", ec="#a5d6a7", lw=0.5, ls="--")
roofs[~roofs.is_candidate].plot(ax=ax, fc="#e53935", ec="#b71c1c", alpha=0.6, lw=0.4)
cand = roofs[roofs.is_candidate]
new = cand.year_built.fillna(0) >= 2017  # not in the 2017 LiDAR
cand[~new].plot(ax=ax, fc="#ff9800", ec="#e65100", alpha=0.65, lw=0.5)
cand[new].plot(ax=ax, fc="#ffeb3b", ec="#f57f17", alpha=0.8, lw=0.8)
ax.legend(handles=[Patch(fc="#ff9800", label=f"roof candidate ({(~new).sum()})"),
                   Patch(fc="#ffeb3b", label=f"roof candidate, built 2017+ ({new.sum()})"),
                   Patch(fc="#e53935", label=f"roof dropped ({(~roofs.is_candidate).sum()})"),
                   Patch(fc="#4caf50", alpha=.5, label=f"open land candidate ({open_.is_candidate.sum()})"),
                   Patch(fc="none", ec="#a5d6a7", ls="--", label="open land screened out")],
          loc="lower left", fontsize=11, framealpha=0.9)
ax.set_axis_off()
ax.set_title("Step 1 screening — TAMU University Buildings and open land", fontsize=14)
fig.tight_layout()
fig.savefig(ROOT / "outputs/step1_screening_map.png")
