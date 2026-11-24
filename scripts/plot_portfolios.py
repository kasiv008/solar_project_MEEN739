"""Map of portfolios over the imagery (outputs/<v>/portfolios/fig_portfolio_maps.png)."""
import re

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from matplotlib.patches import Patch
from rasterio.enums import Resampling

from vintage import ROOT, get_vintage

V = get_vintage()
PD = V["out_dir"] / "portfolios"
INV = V["out_dir"] / "inventory.gpkg"
COL = {"roof": "#2a78d6", "parking": "#eb6834", "open land": "#1baf7a", "garage top": "#eda100"}
SHOW = [("A least land 10%", "A  least land, 10 %"), ("B least cost 10%", "B  least cost, 10 %"),
        ("D grid-limited (export <= 5 %)", "D  grid-limited, export <= 5 %"),
        ("C0 maximum output, no open land", "C0  all roofs + garages + parking")]
slug = lambda n: re.sub(r"[^a-z0-9]+", "_", n.lower().replace("%", "pct")).strip("_")

geo = pd.concat([gpd.read_file(INV, layer="roofs")[["roof_id", "geometry"]].rename(columns={"roof_id": "site_id"}),
                 gpd.read_file(INV, layer="parking_lots")[["lot_id", "geometry"]].rename(columns={"lot_id": "site_id"}),
                 gpd.read_file(INV, layer="open_land")[["open_id", "geometry"]].rename(columns={"open_id": "site_id"})])
summ = pd.read_csv(PD / "summary.csv").set_index("portfolio")
with rasterio.open(ROOT / "tamu_land_resources/tif/merged_tamu_esri_30cm.tif") as src:
    img = src.read([1, 2, 3], out_shape=(3, 1500, 1500), resampling=Resampling.average)
    ext = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
fig, axs = plt.subplots(1, 4, figsize=(20, 6.4), facecolor="#fcfcfb")
for ax, (name, title) in zip(axs, SHOW):
    p = pd.read_csv(PD / f"sites_{slug(name)}.csv")
    ax.imshow(img.transpose(1, 2, 0), extent=ext, alpha=0.55)
    g = geo.merge(p[["site_id", "land_type"]], on="site_id")
    for lt, c in COL.items():
        g[g.land_type == lt].plot(ax=ax, color=c, edgecolor="white", linewidth=0.3)
    s = summ.loc[name]
    ax.set_title(f"{title}\n{s.MWdc:.0f} MWdc · {s.GWh:.0f} GWh/yr · {s.new_land_ha:.0f} ha new land · "
                 f"{s.lcoe_c_kwh:.1f} ¢/kWh", loc="left", fontsize=10, color="#0b0b0b")
    ax.set_axis_off()
fig.legend(handles=[Patch(color=c, label=l) for l, c in COL.items()], loc="lower center", ncol=4, frameon=False)
fig.text(0.01, 0.01, "Placeholder costs (NREL/LBNL benchmarks); FY2026 campus use 650 GWh; 2025 LiDAR inventory.",
         fontsize=8, color="#52514e")
fig.tight_layout(rect=(0, 0.06, 1, 0.97)); fig.savefig(PD / "fig_portfolio_maps.png", dpi=160)
