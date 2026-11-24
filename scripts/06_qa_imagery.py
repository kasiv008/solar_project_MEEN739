"""Step 4: verification against the LiDAR and the 30 cm imagery.

1. Footprint vs LiDAR (automatic): IoU between each TAMU footprint and the LiDAR roof mask
   (nDSM > 2 m), and the shift (+-3 m) that maximises it. Low IoU = footprint outdated or
   building changed; large shift = outline misregistered.
2. Imagery review pack (manual): chips of a stratified sample of roofs -- the largest sites,
   every roof the LiDAR cannot see, every rough/flagged roof, and a random draw of the rest --
   with the footprint, usable cells and shade loss over the imagery, plus a sheet to fill in.

Run per LiDAR vintage:  python 06_qa_imagery.py --vintage 2017|2025
Outputs: outputs/<v>/qa/roof_qa_auto.csv, roof_review_chips.pdf,
         roof_review_sheet.csv (one row per chip; fill the verdict columns)
"""
from pathlib import Path

import warnings

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.backends.backend_pdf import PdfPages
from rasterio import features
from rasterio.windows import from_bounds

from vintage import get_vintage

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
V = get_vintage()
INV = V["out_dir"] / "inventory.gpkg"
LID = V["lidar_dir"]
TIF = ROOT / "tamu_land_resources/tif/merged_tamu_esri_30cm.tif"
QA = V["out_dir"] / "qa"
RES = 0.5
MAX_SHIFT = 6          # cells (3 m)
N_LARGEST, N_RANDOM = 25, 25
SEED = 439
SHIFTS = sorted(((dr, dc) for dr in range(-MAX_SHIFT, MAX_SHIFT + 1) for dc in range(-MAX_SHIFT, MAX_SHIFT + 1)),
                key=lambda t: np.hypot(*t))


def alignment(roofs):
    dtm = rasterio.open(LID / "dtm.tif")
    dsm = rasterio.open(LID / ("dsm_bldg.tif" if (LID / "dsm_bldg.tif").exists() else "dsm_last.tif"))
    rows = []
    for r in roofs.itertuples():
        minx, miny, maxx, maxy = r.geometry.bounds
        pad = MAX_SHIFT * RES + 2
        win = from_bounds(minx - pad, miny - pad, maxx + pad, maxy + pad, dtm.transform).round_offsets().round_lengths()
        tr = dtm.window_transform(win)
        nd = dsm.read(1, window=win, boundless=True) - dtm.read(1, window=win, boundless=True)
        roofm = nd > 2.0
        fp = features.rasterize([(r.geometry, 1)], out_shape=roofm.shape, transform=tr, dtype=np.uint8) == 1
        near_roof = roofm & ndimage_dilate(fp)
        best = (0.0, 0, 0)
        # nearest shifts first; a farther shift must improve IoU (ties keep the smaller shift)
        for dr, dc in SHIFTS:
            sh = np.roll(np.roll(fp, dr, 0), dc, 1)
            inter = (sh & roofm).sum()
            iou = inter / max((sh | near_roof).sum(), 1)
            if iou > best[0] + 1e-6:
                best = (iou, dr, dc)
        inter0 = (fp & roofm).sum()
        iou0 = inter0 / max((fp | near_roof).sum(), 1)
        rows.append(dict(roof_id=r.roof_id, iou=round(iou0, 3), iou_best=round(best[0], 3),
                         shift_e_m=best[2] * RES, shift_n_m=-best[1] * RES,
                         shift_m=round(np.hypot(best[1], best[2]) * RES, 2)))
    return pd.DataFrame(rows)


def ndimage_dilate(m, k=8):
    """Footprint neighbourhood (4 m) so roof cells of *other* buildings do not count against IoU."""
    from scipy import ndimage
    return ndimage.binary_dilation(m, np.ones((2 * k + 1, 2 * k + 1)))


def chips(sample, roofs):
    img = rasterio.open(TIF)
    loss = rasterio.open(LID / "shade_loss.tif")
    secidx = rasterio.open(LID / "usable_roof_section_idx.tif")
    per_page = 4
    with PdfPages(QA / "roof_review_chips.pdf") as pdf:
        for p in range(0, len(sample), per_page):
            fig, axs = plt.subplots(per_page, 3, figsize=(16, 5.2 * per_page), squeeze=False)
            for ax, (_, s) in zip(axs, sample.iloc[p:p + per_page].iterrows()):
                r = roofs.loc[s.roof_id]
                minx, miny, maxx, maxy = r.geometry.buffer(15).bounds
                ext = [minx, maxx, miny, maxy]
                rgb = img.read([1, 2, 3], window=from_bounds(minx, miny, maxx, maxy, img.transform))
                us = secidx.read(1, window=from_bounds(minx, miny, maxx, maxy, secidx.transform)) > 0
                ls = loss.read(1, window=from_bounds(minx, miny, maxx, maxy, loss.transform))
                for a in ax:
                    a.imshow(rgb.transpose(1, 2, 0), extent=ext)
                ax[1].imshow(np.ma.masked_where(~us, us), extent=ext, cmap="autumn", alpha=.5)
                im = ax[2].imshow(np.ma.masked_invalid(ls) * 100, extent=ext, cmap="turbo", vmin=0, vmax=40, alpha=.8)
                for a in ax:
                    gpd.GeoSeries([r.geometry]).boundary.plot(ax=a, color="cyan", lw=1.2)
                    a.set_xlim(minx, maxx); a.set_ylim(miny, maxy); a.set_xticks([]); a.set_yticks([])
                ax[0].set_title(f"{s.roof_id} [{s.stratum}]  {str(r['name'])[:40]}  ({r.year_built if pd.notna(r.year_built) else '?'})",
                                fontsize=10, loc="left")
                ax[1].set_title(f"usable {r.final_usable_m2:,.0f} of {r.area_m2:,.0f} m2 -> {r.system_capacity_kw:,.0f} kWdc"
                                f"  [{r.usable_source.split(' -')[0]}]", fontsize=9, loc="left")
                ax[2].set_title(f"shade loss (kept {r.shade_loss_pct if pd.notna(r.shade_loss_pct) else '-'} %)  "
                                f"IoU {s.iou}  shift {s.shift_m} m  {r.qa_flags or ''}", fontsize=9, loc="left")
            for a in axs[len(sample.iloc[p:p + per_page]):].flat:
                a.set_axis_off()
            fig.tight_layout()
            pdf.savefig(fig, dpi=60)
            plt.close(fig)


def main():
    QA.mkdir(parents=True, exist_ok=True)
    roofs = gpd.read_file(INV, layer="roofs")
    # idempotent: drop results of an earlier QA run
    roofs = roofs.drop(columns=["iou", "iou_best", "shift_e_m", "shift_n_m", "shift_m"], errors="ignore")
    roofs["qa_flags"] = roofs["qa_flags"].fillna("").str.replace(r"(; )?footprint/LiDAR mismatch", "", regex=True)
    auto = alignment(roofs)
    roofs = roofs.merge(auto, on="roof_id")
    roofs["qa_flags"] = roofs["qa_flags"].fillna("")
    # mismatch: poor overlap, or a shift that clearly improves the fit (not just noise)
    misfit = roofs["lidar_ok"] & ((roofs["iou"] < 0.6) |
                                  ((roofs["shift_m"] >= 1.5) & (roofs["iou_best"] - roofs["iou"] > 0.05)))
    roofs.loc[misfit, "qa_flags"] = (roofs.loc[misfit, "qa_flags"] + "; footprint/LiDAR mismatch").str.lstrip("; ")
    roofs.drop(columns="geometry").to_csv(QA / "roof_qa_auto.csv", index=False)
    # write the QA columns back into the inventory
    roofs.to_file(INV, layer="roofs")
    roofs.drop(columns="geometry").to_csv(V["out_dir"] / "roofs.csv", index=False)

    lid = roofs[roofs.lidar_ok]
    print(f"[{V['name']}] footprint vs LiDAR roof (n={len(lid)}): median IoU {lid.iou.median():.2f}, "
          f"median best shift {lid.shift_m.median():.1f} m; {misfit.sum()} flagged mismatch")

    # ---- stratified review sample ----------------------------------------------------------
    rng = np.random.default_rng(SEED)
    roofs = roofs.set_index("roof_id", drop=False)
    strata = []
    largest = roofs.sort_values("system_capacity_kw", ascending=False).head(N_LARGEST).roof_id
    strata += [(i, "largest") for i in largest]
    est = roofs[~roofs.lidar_ok].roof_id
    strata += [(i, "no LiDAR roof") for i in est if i not in set(largest)]
    flagged = roofs[roofs.qa_flags.str.contains("rough|mismatch")].roof_id
    taken = {i for i, _ in strata}
    strata += [(i, "flagged") for i in flagged if i not in taken]
    taken = {i for i, _ in strata}
    rest = [i for i in roofs.roof_id if i not in taken]
    strata += [(i, "random") for i in rng.choice(rest, size=min(N_RANDOM, len(rest)), replace=False)]
    sample = pd.DataFrame(strata, columns=["roof_id", "stratum"]).merge(
        roofs[["roof_id", "name", "iou", "shift_m"]].reset_index(drop=True), on="roof_id")
    chips(sample, roofs)

    sheet = sample[["roof_id", "stratum", "name"]].copy()
    for c in ["footprint_ok (y/n)", "roof_as_in_imagery (y/n/changed)", "usable_area_plausible (y/too high/too low)",
              "obstructions_missed (y/n)", "notes"]:
        sheet[c] = ""
    sheet.to_csv(QA / "roof_review_sheet.csv", index=False)
    print(f"review pack: {len(sample)} roofs -> {sample.stratum.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
