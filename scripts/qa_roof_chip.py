"""Diagnostic chips for single roofs: imagery | nDSM | roughness | tilt | usable.

    python qa_roof_chip.py out.png R0002 R0004 ... [--vintage 2025]
"""
import importlib.util
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rg", ROOT / "scripts/04_roof_geometry.py")
rg = importlib.util.module_from_spec(spec); spec.loader.exec_module(rg)
TIF = ROOT / "tamu_land_resources/tif/merged_tamu_esri_30cm.tif"


def chip(roof_ids, out):
    R, tr = rg.load_surfaces()
    roofs = gpd.read_file(rg.STEP1, layer="roofs_candidates").set_index("roof_id")
    img = rasterio.open(TIF)
    fig, axs = plt.subplots(len(roof_ids), 5, figsize=(22, 4.6 * len(roof_ids)), squeeze=False)
    for ax, rid in zip(axs, roof_ids):
        row = roofs.loc[rid]
        row = row.copy(); row["roof_id"] = rid
        r = type("R", (), dict(row))
        st, secs, usable, _, (sl, wt) = rg.analyse_roof(r, R, tr)
        zl = R["roof"][sl].astype(float); zg = R["dtm"][sl]
        b, c, rms = rg.plane_fit(zl); tilt, asp = rg.tilt_aspect(b, c)
        h, w = zl.shape
        bounds = rasterio.transform.array_bounds(h, w, wt)
        ext = [bounds[0], bounds[2], bounds[1], bounds[3]]
        rgb = img.read([1, 2, 3], window=from_bounds(*bounds, img.transform))
        ax[0].imshow(rgb.transpose(1, 2, 0), extent=ext)
        ax[1].imshow(zl - zg, extent=ext, cmap="viridis", vmin=0, vmax=np.nanpercentile(zl - zg, 99))
        im = ax[2].imshow(rms, extent=ext, cmap="magma", vmin=0, vmax=0.3); plt.colorbar(im, ax=ax[2], shrink=.7)
        im = ax[3].imshow(tilt, extent=ext, cmap="coolwarm", vmin=0, vmax=40); plt.colorbar(im, ax=ax[3], shrink=.7)
        ax[4].imshow(rgb.transpose(1, 2, 0), extent=ext)
        ax[4].imshow(np.ma.masked_where(~usable, usable), extent=ext, cmap="autumn", alpha=.55)
        for a in ax:
            gpd.GeoSeries([row.geometry]).boundary.plot(ax=a, color="cyan", lw=1)
            a.set_xticks([]); a.set_yticks([])
        ax[0].set_title(f"{rid} {str(row['name'])[:35]}", fontsize=10)
        ax[1].set_title("height above ground"); ax[2].set_title("plane-fit RMS [m]")
        ax[3].set_title("tilt [deg]")
        ax[4].set_title(f"usable {st['usable_lidar_m2']:.0f} / {row['area_m2']:.0f} m2  "
                        f"flat {st['flat_frac']:.0%} rough {st['rough_frac']:.0%}", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=60)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--") and a not in ("2017", "2025")]
    chip(args[1:], args[0])
