"""Step 2: roof geometry from LiDAR for every candidate roof, plus terrain/canopy for open land.

Per roof (0.5 m cells inside the TAMU footprint):
  1. roof cells     : nDSM (last returns) > 2 m
  2. local plane    : least-squares plane in a 2.5 m window -> tilt, aspect, RMS roughness
  3. cell classes   : flat (< 10 deg), usable pitch (10-45 deg facing E..S..W), rough (equipment,
                      parapets, edges), tree over roof (all-return surface > last-return surface)
  4. segments       : connected flat / same-facing pitched cells; segments < 20 m2 count as
                      equipment tops; each segment gets its own plane fit (tilt, azimuth)
  5. usable area    : flat + usable pitch, minus trees, 1.2 m from the roof edge (fire access),
                      0.5 m from equipment, strips narrower than 1.5 m and patches < 10 m2 removed
Roofs the 2017 LiDAR does not see (built later) get an estimate from similar roofs, flagged.

Run per LiDAR vintage:  python 04_roof_geometry.py --vintage 2017|2025
Roof surface: building-class points when the vintage has them (2025), else highest last return.

Outputs: data/interim/<v>/inventory_step2.gpkg (roofs, open_land), roof_sections_step2.csv,
         data/interim/lidar_<v>/usable_roof_section_idx.tif, usable_open_id.tif (shading step input)
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.transform import rowcol
from scipy import ndimage

from vintage import get_vintage

ROOT = Path(__file__).resolve().parents[1]
STEP1 = ROOT / "data/interim/inventory_step1.gpkg"
V = get_vintage()
LID = V["lidar_dir"]
OUT = V["interim_dir"] / "inventory_step2.gpkg"
RES = 0.5
CELL = RES * RES

# ---- thresholds (all in one place so the report can quote them) ----------------------------
MIN_ROOF_HEIGHT = 2.0      # m above ground
WIN = 5                    # plane-fit window, cells (2.5 m)
PLANAR_RMS = 0.12          # m; above this a cell is rough (equipment, edge, step, noise)
OBSTACLE_DZ = 0.3          # m above/below the local roof level that makes a cell an obstacle
OBSTACLE_REF_WIN = 9       # cells (4.5 m) median window for the local roof level
FLAT_MAX_TILT = 10.0       # deg
PITCH_MAX_TILT = 45.0      # deg
PITCH_AZ_RANGE = (90, 270) # usable pitched faces: east through south to west
TREE_DZ = 1.5              # m, all-return surface above last-return surface
MIN_SEGMENT_M2 = 20.0      # smaller planar patches are equipment tops / small decks
EDGE_SETBACK = 1.2         # m from the roof outline (IFC fire-access pathways, ~4 ft)
OBSTACLE_CLEARANCE = 0.5   # m around equipment / rough cells
MIN_STRIP = 1.5            # m, narrower strips cannot take a module row
MIN_PATCH_M2 = 10.0        # smaller usable patches dropped
LIDAR_COVER_MIN = 0.5      # share of footprint that must be roof in the LiDAR
LIDAR_YEAR = V["year"]     # buildings completed this year or later are not (fully) in the survey
ROUGH_FLAG = 0.65          # rough share above which a roof is sent to the imagery check

# open land
CANOPY_HEIGHT = 2.0        # m
OPEN_OBSTACLE_BUFFER = 2.0 # m around trees / poles / structures
OPEN_MAX_SLOPE = 8.0       # deg (fixed racks and trackers on campus-grade land)


def load(name):
    with rasterio.open(LID / f"{name}.tif") as src:
        return src.read(1), src.transform


def load_surfaces():
    """dtm, dsm_all and the roof surface (building class if present, else last returns)."""
    R, tr = {}, None
    for name in ["dtm", "dsm_all"]:
        R[name], tr = load(name)
    roof_src = "dsm_bldg" if (LID / "dsm_bldg.tif").exists() else "dsm_last"
    R["roof"], _ = load(roof_src)
    R["roof_src"] = roof_src
    return R, tr


def plane_fit(z):
    """Least-squares plane in a WIN x WIN window for every cell (separable sums).
    Returns slope components dz/dx (east), dz/dy (north) and the RMS residual."""
    h = WIN // 2
    off = np.arange(-h, h + 1) * RES
    ones = np.ones(WIN)
    sxx = WIN * (off**2).sum()
    zc = z - np.nanmean(z)  # numerical stability
    def corr(a, kr, kc):
        return ndimage.correlate1d(ndimage.correlate1d(a, kr, axis=0, mode="nearest"), kc,
                                   axis=1, mode="nearest")
    n = WIN * WIN
    s = corr(zc, ones, ones)
    sx = corr(zc, ones, off)           # columns -> +x (east)
    sy = corr(zc, -off, ones)          # rows go south -> +y (north) is -row
    s2 = corr(zc * zc, ones, ones)
    a, b, c = s / n, sx / sxx, sy / sxx
    sse = np.maximum(s2 - n * a * a - b * b * sxx - c * c * sxx, 0)
    return b, c, np.sqrt(sse / n)


def tilt_aspect(b, c):
    tilt = np.degrees(np.arctan(np.hypot(b, c)))
    aspect = (np.degrees(np.arctan2(-b, -c)) + 360) % 360  # downslope direction, from north
    return tilt, aspect


def fit_segment(z, rr, cc, transform):
    """Plane fit over a segment's cells -> tilt, azimuth (facing), rms."""
    x = transform.c + (cc + 0.5) * RES
    y = transform.f - (rr + 0.5) * RES
    A = np.c_[x - x.mean(), y - y.mean(), np.ones_like(x)]
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    t, a = tilt_aspect(coef[0], coef[1])
    rms = np.sqrt(np.mean((A @ coef - z) ** 2))
    return float(t), float(a), float(rms)


def building_axis(geom):
    """Long-axis bearing (0-180 deg from north) of the minimum rotated rectangle."""
    mrr = geom.minimum_rotated_rectangle
    xs, ys = mrr.exterior.coords.xy
    e = [(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(2)]
    dx, dy = max(e, key=lambda v: np.hypot(*v))
    return (np.degrees(np.arctan2(dx, dy)) + 180) % 180


def edge_aligned_azimuth(axis):
    """Row azimuth when rows run parallel to a building edge: the edge normal closest to south."""
    cands = [(axis + k * 90) % 360 for k in range(4)]
    return min(cands, key=lambda a: abs(a - 180))


def analyse_roof(row, R, transform):
    dsm_l, dsm_a, dtm = R["roof"], R["dsm_all"], R["dtm"]
    minx, miny, maxx, maxy = row.geometry.bounds
    r0, c0 = rowcol(transform, minx - 5, maxy + 5)
    r1, c1 = rowcol(transform, maxx + 5, miny - 5)
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, dsm_l.shape[0] - 1), min(c1, dsm_l.shape[1] - 1)
    sl = (slice(r0, r1 + 1), slice(c0, c1 + 1))
    wt = rasterio.windows.transform(rasterio.windows.Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1), transform)
    zl, za, zg = dsm_l[sl].astype(np.float64), dsm_a[sl], dtm[sl]
    fp = features.rasterize([(row.geometry, 1)], out_shape=zl.shape, transform=wt, dtype=np.uint8) == 1

    nd = zl - zg
    roof = fp & (nd > MIN_ROOF_HEIGHT)
    b, c, rms = plane_fit(zl)
    tilt, aspect = tilt_aspect(b, c)
    tree = ndimage.binary_opening(fp & ((za - zl) > TREE_DZ), np.ones((2, 2)))
    planar = rms < PLANAR_RMS
    # obstacles need a real height difference from the surrounding roof (4.5 m median):
    # vents, fans, poles, steps, drains / openings. Rough cells within +-0.3 m (joints, curbs,
    # single noisy points) are not obstacles - racking spans them.
    ref = ndimage.median_filter(zl, size=OBSTACLE_REF_WIN)
    obstacle = roof & (np.abs(zl - ref) > OBSTACLE_DZ)
    tilt_ref, _ = tilt_aspect(*plane_fit(ref)[:2])
    flat = roof & ~obstacle & (np.where(planar, tilt, tilt_ref) < FLAT_MAX_TILT)
    facing_ok = (aspect >= PITCH_AZ_RANGE[0]) & (aspect <= PITCH_AZ_RANGE[1])
    pitched = roof & ~obstacle & planar & (tilt >= FLAT_MAX_TILT) & (tilt <= PITCH_MAX_TILT)
    rough = roof & ~(flat | pitched)

    # segments: flat = class 0, pitched by facing octant 1..8
    cls = np.full(zl.shape, -1, np.int16)
    cls[flat] = 0
    octant = (((aspect + 22.5) % 360) // 45).astype(np.int16) + 1
    cls[pitched] = octant[pitched]
    seg = np.zeros(zl.shape, np.int32)
    nseg = 0
    for k in np.unique(cls[cls >= 0]):
        lab, n = ndimage.label(cls == k, structure=np.ones((3, 3)))
        seg[lab > 0] = lab[lab > 0] + nseg
        nseg += n
    sizes = np.bincount(seg.ravel(), minlength=nseg + 1) * CELL
    small = (sizes < MIN_SEGMENT_M2)[seg] & (seg > 0)
    equipment = obstacle | small
    seg[small] = 0

    # usable
    ok_seg = np.zeros(nseg + 1, bool)
    seg_rows = []
    for sid in np.unique(seg[seg > 0]):
        rr, cc = np.nonzero(seg == sid)
        t, a, r = fit_segment(zl[rr, cc], rr, cc, wt)
        k = int(cls[rr[0], cc[0]])
        usable_face = (t < FLAT_MAX_TILT) or (t <= PITCH_MAX_TILT and PITCH_AZ_RANGE[0] <= a <= PITCH_AZ_RANGE[1])
        ok_seg[sid] = usable_face
        seg_rows.append(dict(seg=int(sid), kind="flat" if k == 0 else "pitched", seg_tilt=round(t, 1),
                             seg_azimuth=round(a, 1), seg_rms=round(r, 3), seg_plan_m2=len(rr) * CELL,
                             seg_height_m=float(np.median(nd[rr, cc]))))
    usable = ok_seg[seg] & ~tree
    usable &= ndimage.distance_transform_edt(roof) * RES >= EDGE_SETBACK
    usable &= ndimage.distance_transform_edt(~equipment) * RES > OBSTACLE_CLEARANCE
    k = int(np.ceil(MIN_STRIP / RES))
    usable = ndimage.binary_opening(usable, np.ones((k, k)))
    lab, n = ndimage.label(usable)
    if n:
        keep = np.bincount(lab.ravel())[1:] * CELL >= MIN_PATCH_M2
        usable = np.isin(lab, np.nonzero(keep)[0] + 1)

    nfp = fp.sum()
    useful_roof = roof.sum()
    stats = dict(
        fp_cells_m2=nfp * CELL,
        lidar_cover=round(useful_roof / max(nfp, 1), 3),
        roof_m2=useful_roof * CELL,
        height_m=round(float(np.median(nd[roof])), 1) if useful_roof else np.nan,
        ground_elev_m=round(float(np.median(zg[fp])), 1),
        flat_frac=round(flat.sum() / max(useful_roof, 1), 3),
        pitched_frac=round(pitched.sum() / max(useful_roof, 1), 3),
        pitched_ok_frac=round((pitched & facing_ok).sum() / max(useful_roof, 1), 3),
        obstacle_frac=round((obstacle & roof).sum() / max(useful_roof, 1), 3),
        rough_frac=round((equipment & roof).sum() / max(useful_roof, 1), 3),
        tree_frac=round((tree & roof).sum() / max(useful_roof, 1), 3),
        median_tilt=round(float(np.median(tilt[roof & planar])), 1) if (roof & planar).any() else np.nan,
        usable_lidar_m2=usable.sum() * CELL,
    )
    # sections = usable cells grouped by segment
    secs = []
    segd = {s["seg"]: s for s in seg_rows}
    for sid in np.unique(seg[usable]):
        m = usable & (seg == sid)
        s = segd[int(sid)]
        plan = m.sum() * CELL
        secs.append({**s, "usable_plan_m2": plan,
                     "usable_surface_m2": plan / np.cos(np.radians(s["seg_tilt"]))})
    return stats, secs, usable, seg, (sl, wt)


def roofs_step(roofs, R, transform):
    """Returns per-roof stats, per-section table and a raster of section index (1-based,
    row number in the section table + 1) for every usable cell, used by the shading step."""
    section_idx = np.zeros(R["dtm"].shape, np.int32)
    rows, sections = [], []
    for row in roofs.itertuples():
        stats, secs, usable, seg, (sl, _) = analyse_roof(row, R, transform)
        rows.append({"roof_id": row.roof_id, **stats})
        sub = section_idx[sl]
        for j, s in enumerate(sorted(secs, key=lambda d: -d["usable_plan_m2"]), 1):
            sections.append({"roof_id": row.roof_id, "section_id": f"{row.roof_id}-S{j:02d}", **s})
            sub[usable & (seg == s["seg"]) & (sub == 0)] = len(sections)
    return pd.DataFrame(rows), pd.DataFrame(sections), section_idx


def open_land_step(open_, R, transform):
    dtm, dsm_a = R["dtm"], R["dsm_all"]
    nd = dsm_a - dtm
    gb, gc, _ = plane_fit(dtm.astype(np.float64))
    slope, gaspect = tilt_aspect(gb, gc)
    canopy = nd > CANOPY_HEIGHT
    obstacle = nd > 1.0  # trees, poles, small structures, parked equipment
    clear = ndimage.distance_transform_edt(~obstacle) * RES > OPEN_OBSTACLE_BUFFER
    ids = features.rasterize([(g, i + 1) for i, g in enumerate(open_.geometry)],
                             out_shape=dtm.shape, transform=transform, dtype=np.int32)
    usable = (ids > 0) & clear & (slope <= OPEN_MAX_SLOPE)
    k = int(np.ceil(5 / RES))  # arrays need room: drop strips < 5 m wide
    usable = ndimage.binary_opening(usable, np.ones((k, k)))
    n = len(open_) + 1
    cnt = lambda m: np.bincount(ids[m], minlength=n)[1:]
    tot = np.maximum(cnt(ids > 0), 1)
    within = ids > 0
    df = pd.DataFrame({
        "open_id": open_["open_id"].values,
        "canopy_frac": (cnt(within & canopy) / tot).round(3),
        "slope_mean_deg": (np.bincount(ids[within], slope[within], n)[1:] / tot).round(1),
        "slope_p90_deg": [round(float(np.percentile(slope[ids == i + 1], 90)), 1) if (ids == i + 1).any() else np.nan
                          for i in range(len(open_))],
        "ground_elev_m": (np.bincount(ids[within], dtm[within], n)[1:] / tot).round(1),
        "usable_lidar_m2": cnt(usable) * CELL,
    })
    # mean downslope direction of the usable part (matters for tracker axis / fixed tilt)
    ux = np.bincount(ids[usable], np.sin(np.radians(gaspect[usable])), n)[1:]
    uy = np.bincount(ids[usable], np.cos(np.radians(gaspect[usable])), n)[1:]
    df["ground_aspect_deg"] = ((np.degrees(np.arctan2(ux, uy)) + 360) % 360).round(0)
    usable_id = np.where(usable, ids, 0).astype(np.int32)
    return df, usable_id


def main():
    R, transform = load_surfaces()
    print(f"vintage {V['name']}: roof surface from {R['roof_src']}")
    roofs = gpd.read_file(STEP1, layer="roofs_candidates")
    open_ = gpd.read_file(STEP1, layer="open_land_candidates")

    stats, sections, usable_roof = roofs_step(roofs, R, transform)
    roofs = roofs.merge(stats, on="roof_id")
    roofs["bldg_axis_deg"] = [round(building_axis(g), 1) for g in roofs.geometry]
    roofs["row_azimuth_edge"] = roofs["bldg_axis_deg"].map(edge_aligned_azimuth).round(1)

    # roofs the LiDAR cannot see -> estimate from LiDAR-measured roofs of the same type
    # built 2017+ -> the LiDAR shows whatever stood there before (e.g. Coolidge Center, 2023)
    newer = roofs["year_built"].fillna(0) >= LIDAR_YEAR
    roofs["lidar_ok"] = (roofs["lidar_cover"] >= LIDAR_COVER_MIN) & ~newer
    roofs["usable_ratio"] = roofs["usable_lidar_m2"] / roofs["area_m2"]
    flat_like = roofs["lidar_ok"] & (roofs["flat_frac"] >= 0.5)
    by_type = roofs[flat_like].groupby("tamu_type")["usable_ratio"].median()
    overall = roofs.loc[flat_like, "usable_ratio"].median()
    est = roofs["tamu_type"].map(by_type).fillna(overall)
    roofs["usable_m2"] = np.where(roofs["lidar_ok"], roofs["usable_lidar_m2"], roofs["area_m2"] * est).round(1)
    roofs["usable_source"] = np.select(
        [newer, ~roofs["lidar_ok"]],
        [f"estimate: built {LIDAR_YEAR}+ - check imagery", f"estimate: no roof in {LIDAR_YEAR} LiDAR - check imagery"],
        default=f"LiDAR {LIDAR_YEAR}")
    roofs.loc[newer, "qa_flags"] = (roofs.loc[newer, "qa_flags"].fillna("") +
                                    f"; built {LIDAR_YEAR}+ (not in LiDAR)").str.lstrip("; ")
    # very rough LiDAR roofs: heavy equipment, or under construction / renovation in Jan 2017
    rough = roofs["lidar_ok"] & (roofs["rough_frac"] > ROUGH_FLAG)
    roofs.loc[rough, "qa_flags"] = (roofs.loc[rough, "qa_flags"].fillna("") +
                                    "; rough roof in LiDAR - check imagery").str.lstrip("; ")
    print(f"usable/footprint, flat LiDAR roofs: median {overall:.2f}; by type:\n{by_type.round(2).to_string()}")

    roofs.to_file(OUT, layer="roofs")
    sections.to_csv(V["interim_dir"] / "roof_sections_step2.csv", index=False)

    open_ = gpd.read_file(STEP1, layer="open_land_candidates")
    ostats, usable_open = open_land_step(open_, R, transform)
    open_ = open_.merge(ostats, on="open_id")
    open_.to_file(OUT, layer="open_land")

    prof = dict(driver="GTiff", height=usable_roof.shape[0], width=usable_roof.shape[1], count=1,
                dtype="int32", crs="EPSG:6343", transform=transform, nodata=0, compress="deflate")
    with rasterio.open(LID / "usable_roof_section_idx.tif", "w", **prof) as dst:
        dst.write(usable_roof, 1)
    with rasterio.open(LID / "usable_open_id.tif", "w", **prof) as dst:
        dst.write(usable_open, 1)

    # ---- summary ------------------------------------------------------------------------
    ok = roofs["lidar_ok"]
    print(f"\nROOFS: {len(roofs)} candidates; LiDAR roof found for {ok.sum()}, missing for {(~ok).sum()}")
    print(f"footprint {roofs.area_m2.sum():,.0f} m2 -> usable {roofs.usable_m2.sum():,.0f} m2 "
          f"({roofs.usable_m2.sum() / roofs.area_m2.sum():.0%}); sections: {len(sections)} "
          f"({(sections.kind == 'pitched').sum()} pitched)")
    print(roofs.loc[ok, ["flat_frac", "pitched_ok_frac", "rough_frac", "tree_frac", "usable_ratio",
                         "height_m"]].describe().round(2).to_string())
    print(f"\nOPEN LAND: {len(open_)} tracts, {open_.area_m2.sum() / 1e4:.1f} ha -> usable "
          f"{open_.usable_lidar_m2.sum() / 1e4:.1f} ha; median canopy {open_.canopy_frac.median():.0%}")


if __name__ == "__main__":
    main()
