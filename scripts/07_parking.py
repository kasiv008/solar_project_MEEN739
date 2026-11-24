"""Step 5: surface-parking carport inventory (canopies over the striped stall rows).

  python 07_parking.py --vintage 2017|2025

Lots: TAMU "Surface Parking" layer. Kept: Surface / Visitor / Restricted / Misc lots, >= 2,000 m2,
mostly in the tile. Dropped (with reason): private, on-street, city, loading zones.

Canopies: every painted stall divider (TAMU "Parking Line", use = Space, ~5.2 m long) is widened by
half a stall width on each side; the union gives the stall rows a carport would cover (double-loaded
rows merge into ~10 m wide strips, the usual T-canopy). Drive aisles stay open.
LiDAR (vintage): strip cells under trees (> 2.5 m above ground, +1 m clearance) are removed; the
rest get the same horizon / TMY shading as the roofs, from canopy height; > 20 % loss dropped.
Lots without stall lines get an estimate (median canopy share of striped lots), flagged.

Each lot x canopy orientation is one PVWatts system (fixed open rack, 7 deg, facing the strip
normal closest to south) with shading_azal / shading_diff.

Outputs: outputs/<v>/parking_lots.csv, parking_sections.csv,
         outputs/<v>/inventory.gpkg layers parking_lots, parking_canopies
"""
import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from scipy import ndimage
from shapely import box

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sh", ROOT / "scripts/05_shading.py")
sh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sh)  # horizons, shade_loss, azal_table, tmy, obstacle_grid, CFG, V

V, CFG = sh.V, sh.CFG
RAW = ROOT / "data/raw/tamu_fcor.gpkg"
LID = V["lidar_dir"]
OUTD = V["out_dir"]
TILE = box(753000, 3388500, 756000, 3391500)
X0, Y1, RES = 753000.0, 3391500.0, 0.5

KEEP_TYPES = {"Surface", "Surface Visitor", "Surface Restricted", "Surface Misc"}
DROP_TYPES = {"Surface Private": "private lot", "Street": "on-street parking",
              "CitArea": "city area", "Surface Loading": "loading zone (truck clearance)"}
MIN_LOT_M2 = 2000
STALL_HALF_WIDTH = 1.35        # m (9 ft stalls)
STALL_LINE_LEN = (3.0, 12.0)   # m; single dividers ~5.2 m, back-to-back dividers up to ~10.5 m
MIN_STRIP_M2 = 30              # ~2 stalls
TREE_HEIGHT = 2.5              # m above ground
TREE_CLEARANCE = 1.0           # m
CANOPY_HEIGHT = 4.5            # m, module plane above ground (~14 ft clearance + structure)
ARCH = CFG["archetypes"]["carport"]


def lots_screen():
    lots = gpd.read_file(RAW, layer="surface_parking")
    lots = lots.rename(columns={"LotName": "lot_name", "LotType": "lot_type"})[["lot_name", "lot_type", "geometry"]]
    lots["lot_type"] = lots["lot_type"].astype("string").str.strip()
    lots["area_m2"] = lots.area.round(1)
    lots["frac_in_tile"] = (lots.intersection(TILE).area / lots.area).round(3)
    reason = pd.Series(pd.NA, index=lots.index, dtype="string")
    for t, why in DROP_TYPES.items():
        reason[(lots["lot_type"] == t) & reason.isna()] = why
    reason[~lots["lot_type"].isin(KEEP_TYPES | set(DROP_TYPES)) & reason.isna()] = "unknown lot type"
    reason[(lots["frac_in_tile"] < 0.5) & reason.isna()] = "mostly outside the tile"
    reason[(lots["area_m2"] < MIN_LOT_M2) & reason.isna()] = f"too small (< {MIN_LOT_M2} m2)"
    lots["screen_reason"] = reason.fillna("candidate")
    lots["is_candidate"] = reason.isna()
    lots["qa_flags"] = np.where(lots["lot_name"].str.contains("Bus", case=False, na=False),
                                "bus lot - taller canopy clearance", "")
    lots = lots.sort_values("area_m2", ascending=False).reset_index(drop=True)
    lots.insert(0, "lot_id", [f"P{i:03d}" for i in range(1, len(lots) + 1)])
    return lots


def stall_lines(lots):
    pl = gpd.read_file(RAW, layer="parking_line")
    pl = pl[pl["PKG_Use"].eq("Space")].explode(index_parts=False).reset_index(drop=True)
    pl = pl[pl.length.between(*STALL_LINE_LEN)]
    mid = pl.copy()
    mid["geometry"] = pl.interpolate(0.5, normalized=True)
    j = gpd.sjoin(mid[["geometry"]], lots[["lot_id", "geometry"]].assign(geometry=lots.buffer(1.0)),
                  predicate="within")
    j = j[~j.index.duplicated()]  # a divider on the shared edge of two lots goes to one of them
    pl = pl.loc[j.index].assign(lot_id=j["lot_id"].values)
    return pl[["lot_id", "geometry"]]


def canopy_strips(lots, lines):
    rows = []
    for lot_id, g in lines.groupby("lot_id"):
        lot_geom = lots.loc[lots.lot_id == lot_id, "geometry"].iloc[0]
        u = g.buffer(STALL_HALF_WIDTH, cap_style="flat").union_all()
        u = u.buffer(0.5, join_style="mitre").buffer(-0.5, join_style="mitre")  # close hairline gaps
        u = u.intersection(lot_geom.buffer(1.0))
        parts = gpd.GeoSeries([u], crs=lots.crs).explode(index_parts=False)
        for p in parts[parts.area >= MIN_STRIP_M2]:
            rows.append(dict(lot_id=lot_id, n_dividers=int(g.intersects(p).sum()), geometry=p))
    s = gpd.GeoDataFrame(rows, crs=lots.crs)
    s["canopy_m2"] = s.area.round(1)
    axis = [sh_axis(p) for p in s.geometry]
    s["strip_length_m"] = [round(a[1], 1) for a in axis]
    s["strip_width_m"] = (s["canopy_m2"] / s["strip_length_m"]).round(1)
    s["axis_deg"] = [round(a[0], 1) for a in axis]
    s["azimuth"] = s["axis_deg"].map(lambda a: min(((a + k * 90) % 360 for k in range(4)), key=lambda z: abs(z - 180)))
    s["azimuth"] = (s["azimuth"] / 5).round().mul(5)
    s.insert(0, "strip_id", [f"{l}-K{i:03d}" for l, i in zip(s.lot_id, s.groupby("lot_id").cumcount() + 1)])
    return s


def sh_axis(geom):
    """(bearing of the long axis 0-180, long side length) of the minimum rotated rectangle."""
    xs, ys = geom.minimum_rotated_rectangle.exterior.coords.xy
    e = [(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(2)]
    dx, dy = max(e, key=lambda v: np.hypot(*v))
    return (np.degrees(np.arctan2(dx, dy)) + 180) % 180, float(np.hypot(dx, dy))


def lidar_and_shading(strips):
    with rasterio.open(LID / "dtm.tif") as src:
        dtm, tr = src.read(1), src.transform
    with rasterio.open(LID / "dsm_all.tif") as src:
        dsm = src.read(1)
    sid = features.rasterize([(g, i + 1) for i, g in enumerate(strips.geometry)], out_shape=dtm.shape,
                             transform=tr, dtype=np.int32)
    tall = (dsm - dtm) > TREE_HEIGHT
    tree = ndimage.distance_transform_edt(~tall) * RES <= TREE_CLEARANCE
    clear = (sid > 0) & ~tree
    n = len(strips) + 1
    strips["tree_conflict_m2"] = (np.bincount(sid[(sid > 0) & tree], minlength=n)[1:] * RES**2).round(1)
    strips["clear_m2"] = (np.bincount(sid[clear], minlength=n)[1:] * RES**2).round(1)

    # shading from canopy height, 1 m sample of the clear cells
    r, c = np.nonzero(clear[::2, ::2])
    r, c = r * 2, c * 2
    ids = sid[r, c]
    x, y = X0 + (c + 0.5) * RES, Y1 - (r + 0.5) * RES
    z0 = dtm[r, c] + CANOPY_HEIGHT
    print(f"canopy sample points: {len(x):,}")
    H = sh.horizons(x, y, z0, sh.obstacle_grid())
    sun = sh.tmy()
    loss = np.zeros(len(x)); svf = np.zeros(len(x))
    az = strips["azimuth"].values
    for a in np.unique(az):
        m = np.isin(ids, np.nonzero(az == a)[0] + 1)
        if m.any():
            loss[m], svf[m] = sh.shade_loss(H[m], ARCH["tilt"], float(a), sun)
    keep = loss <= sh.MAX_LOSS
    cnt = lambda msk: np.bincount(ids[msk], minlength=n)[1:]
    tot = np.maximum(cnt(np.ones_like(keep)), 1)
    strips["shade_pass_frac"] = np.where(cnt(np.ones_like(keep)) > 0, cnt(keep) / tot, 0).round(3)
    strips["final_canopy_m2"] = (strips["clear_m2"] * strips["shade_pass_frac"]).round(1)
    return strips, dict(ids=ids, H=H, loss=loss, svf=svf, keep=keep)


def sections(strips, cells):
    """One PVWatts system per lot x canopy azimuth."""
    strips = strips.reset_index(drop=True)
    strip_row = cells["ids"] - 1
    out = []
    for (lot_id, az), g in strips.groupby(["lot_id", "azimuth"]):
        m = np.isin(strip_row, g.index.values) & cells["keep"]
        area = g["final_canopy_m2"].sum()
        if area < MIN_STRIP_M2:
            continue
        out.append(dict(
            lot_id=lot_id, section_id=f"{lot_id}-A{int(az):03d}", n_strips=len(g), archetype="carport",
            array_type=ARCH["array_type"], tilt=ARCH["tilt"], azimuth=az, gcr=ARCH["gcr"],
            kw_per_m2=ARCH["kw_per_m2"], final_canopy_m2=round(area, 1),
            system_capacity_kw=round(area * ARCH["kw_per_m2"], 1),
            shade_loss_pct=round(100 * cells["loss"][m].mean(), 1) if m.any() else np.nan,
            shading_diff=round(100 * (1 - cells["svf"][m].mean()), 1) if m.any() else np.nan,
            shading_azal=json.dumps(sh.azal_table(cells["H"][m])) if m.any() else None))
    return pd.DataFrame(out)


def main():
    lots = lots_screen()
    cand = lots[lots.is_candidate].copy()
    lines = stall_lines(cand)
    strips = canopy_strips(cand, lines)
    strips, cells = lidar_and_shading(strips)
    secs = sections(strips, cells)

    agg = strips.groupby("lot_id").agg(n_strips=("strip_id", "count"), n_stalls_approx=("n_dividers", "sum"),
                                       canopy_m2=("canopy_m2", "sum"), tree_conflict_m2=("tree_conflict_m2", "sum"),
                                       canopy_length_m=("strip_length_m", "sum"), final_canopy_m2=("final_canopy_m2", "sum"))
    sagg = secs.assign(w=secs.final_canopy_m2 * secs.shade_loss_pct).groupby("lot_id")
    agg["shade_loss_pct"] = (sagg.w.sum() / sagg.final_canopy_m2.sum()).round(1)
    agg["system_capacity_kw"] = sagg.system_capacity_kw.sum()
    lots = lots.merge(agg, left_on="lot_id", right_index=True, how="left")

    # candidate lots without stall lines -> estimate from striped lots (flagged)
    striped = lots.is_candidate & lots.canopy_m2.notna()
    ratio = (lots.loc[striped, "final_canopy_m2"] / lots.loc[striped, "area_m2"]).median()
    est = lots.is_candidate & ~striped
    lots.loc[est, "final_canopy_m2"] = (lots.loc[est, "area_m2"] * ratio).round(1)
    lots.loc[est, "system_capacity_kw"] = (lots.loc[est, "final_canopy_m2"] * ARCH["kw_per_m2"]).round(1)
    lots["canopy_source"] = np.select([striped, est], ["stall lines + LiDAR " + V["name"],
                                                       "estimate: no stall lines - check imagery"], "")
    lots.loc[est, "qa_flags"] = (lots.loc[est, "qa_flags"] + "; no stall lines").str.lstrip("; ")
    print(f"canopy share of lot area (striped lots, median): {ratio:.2f}")

    lots["archetype"] = np.where(lots.is_candidate, "carport", None)
    for k in ("mount", "structure", "cost_key"):
        lots[k] = np.where(lots.is_candidate, ARCH[k], None)
    usd = CFG["costs"]["installed_usd_per_wdc"][ARCH["cost_key"]]
    lots["usd_per_wdc_placeholder"] = np.where(lots.is_candidate, usd, np.nan)
    lots["capex_musd_placeholder"] = (lots["system_capacity_kw"] * usd / 1e3).round(2)
    lots["dist_substation_m"] = sh.dist_to_substation(lots)
    pt = lots.geometry.representative_point()
    lots["lat"], lots["lon"] = pt.to_crs(4326).y.round(6), pt.to_crs(4326).x.round(6)
    secs = secs.merge(lots[["lot_id", "lot_name", "lat", "lon"]], on="lot_id", how="left")

    inv = OUTD / "inventory.gpkg"
    lots.to_file(inv, layer="parking_lots")
    strips.to_file(inv, layer="parking_canopies")
    lots.drop(columns="geometry").to_csv(OUTD / "parking_lots.csv", index=False)
    secs.to_csv(OUTD / "parking_sections.csv", index=False)

    c = lots[lots.is_candidate]
    print(lots.groupby("screen_reason")["area_m2"].agg(n="count", m2="sum").round(0).to_string())
    print(f"\n[{V['name']} LiDAR] PARKING: {len(c)} lots, {c.area_m2.sum() / 1e4:.1f} ha; stall-row canopy "
          f"{c.canopy_m2.sum():,.0f} m2 (~{int(c.n_stalls_approx.sum()):,} stalls), tree conflict "
          f"{c.tree_conflict_m2.sum():,.0f} m2 -> final {c.final_canopy_m2.sum():,.0f} m2, "
          f"{c.system_capacity_kw.sum() / 1e3:.1f} MWdc ({lots.loc[est, 'system_capacity_kw'].sum() / 1e3:.1f} estimated); "
          f"sections {len(secs)}; weighted shade loss "
          f"{np.average(secs.shade_loss_pct, weights=secs.final_canopy_m2):.1f} %")


if __name__ == "__main__":
    main()
