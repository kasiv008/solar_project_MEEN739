"""Step 3: shading from LiDAR horizons + TMY, and the final PySAM-ready site tables.

For every usable cell (roofs: 1 m sample of the 0.5 m usable mask; open land: 2 m sample) the
horizon angle is traced in 36 directions over the all-return surface (buildings + trees), out
to 300 m. With the College Station TMY this gives, per cell, the share of annual plane-of-array
irradiance lost to shade (beam: hour-by-hour sun vs horizon; diffuse: isotropic sky-view factor).
Cells losing more than 20 % are dropped (NREL rooftop-potential rule).

Per section / tract it then writes the two PVWatts v8 shading inputs:
  shading_azal  azimuth x altitude beam-shading loss table [%] (share of the array's cells that
                are shaded with the sun at that position)
  shading_diff  sky-diffuse shading loss [%] (1 - mean sky-view factor)
plus tilt, azimuth, GCR, array type and kWdc from config/sam_defaults.json.

Run per LiDAR vintage:  python 05_shading.py --vintage 2017|2025
Outputs: outputs/<v>/inventory.gpkg (roofs, open_land), roofs.csv, roof_sections.csv, open_land.csv,
         data/interim/lidar_<v>/shade_loss.tif (annual loss per sampled cell, for maps/QA).
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pvlib
import rasterio

from vintage import get_vintage

ROOT = Path(__file__).resolve().parents[1]
V = get_vintage()
LID = V["lidar_dir"]
STEP2 = V["interim_dir"] / "inventory_step2.gpkg"
OUTD = V["out_dir"]
CFG = json.loads((ROOT / "config/sam_defaults.json").read_text())
OUT = OUTD / "inventory.gpkg"

X0, Y1, RES = 753000.0, 3391500.0, 0.5
N_DIR = 36                                   # horizon directions (10 deg)
DIST = np.unique(np.round(np.geomspace(1.5, 300, 80), 2))  # ray samples, m
PANEL_H_ROOF = 0.3                           # module height above roof deck, m
PANEL_H_GROUND = 1.0                         # mid-module height above ground, m
MAX_LOSS = CFG["screening"]["max_annual_shade_loss"]
AZAL_ALT = np.arange(0, 91, 5)
AZAL_AZ = np.arange(0, 361, 10)
CURRENT_YEAR = 2026


# ---------------------------------------------------------------------------------------------
def load(name):
    with rasterio.open(LID / f"{name}.tif") as src:
        return src.read(1)


def obstacle_grid():
    """All-return surface at 1 m (max of the 0.5 m cells): everything that can cast shade."""
    d = load("dsm_all")
    h, w = d.shape
    return d.reshape(h // 2, 2, w // 2, 2).max(axis=(1, 3))


def horizons(x, y, z0, obs):
    """Horizon elevation angle [deg] per point and direction -> (n, N_DIR) float32."""
    ny, nx = obs.shape
    H = np.full((x.size, N_DIR), -90.0, np.float32)
    for k in range(N_DIR):
        az = np.radians(k * 360 / N_DIR)
        dx, dy = np.sin(az), np.cos(az)
        best = np.full(x.size, -np.inf, np.float32)
        for d in DIST:
            c = ((x + d * dx - X0) / 1.0).astype(np.int32)
            r = ((Y1 - (y + d * dy)) / 1.0).astype(np.int32)
            ok = (c >= 0) & (c < nx) & (r >= 0) & (r < ny)
            dz = np.where(ok, obs[np.clip(r, 0, ny - 1), np.clip(c, 0, nx - 1)] - z0, -np.inf)
            np.maximum(best, dz / d, out=best)  # tan(angle) is monotonic in angle
        H[:, k] = np.degrees(np.arctan(best))
    return H


def tmy():
    path = ROOT / CFG["weather_file"]
    meta = pd.read_csv(path, nrows=1)
    w = pd.read_csv(path, skiprows=2)
    t = pd.to_datetime(dict(year=2023, month=w.Month, day=w.Day, hour=w.Hour, minute=w.Minute))
    t = t.dt.tz_localize(f"Etc/GMT{-int(meta['Time Zone'][0]):+d}")
    sp = pvlib.solarposition.get_solarposition(t, float(meta.Latitude[0]), float(meta.Longitude[0]))
    df = pd.DataFrame({"dni": w.DNI.values, "dhi": w.DHI.values, "zen": sp.apparent_zenith.values,
                       "az": sp.azimuth.values, "alt": sp.apparent_elevation.values})
    return df[(df.alt > 0) & ((df.dni > 0) | (df.dhi > 0))].reset_index(drop=True)


def shade_loss(H, tilt, azim, sun):
    """Annual POA shading loss per cell for one array orientation, and the sky-view factor."""
    cos_aoi = np.cos(np.radians(pvlib.irradiance.aoi(tilt, azim, sun.zen.values, sun.az.values)))
    beam = sun.dni.values * np.clip(cos_aoi, 0, None)
    diff = sun.dhi.values * (1 + np.cos(np.radians(tilt))) / 2
    kbin = np.round(sun.az.values / (360 / N_DIR)).astype(int) % N_DIR
    shaded_beam = np.zeros(H.shape[0])
    for k in range(N_DIR):
        m = kbin == k
        if not m.any():
            continue
        order = np.argsort(sun.alt.values[m])
        alts, cw = sun.alt.values[m][order], np.cumsum(beam[m][order])
        # beam lost = weight of the hours in this sector whose sun altitude is below the horizon
        idx = np.searchsorted(alts, H[:, k], side="left")
        shaded_beam += np.where(idx > 0, cw[np.maximum(idx - 1, 0)], 0)
    svf = np.mean(np.cos(np.radians(np.clip(H, 0, 90))) ** 2, axis=1)
    loss = (shaded_beam + diff.sum() * (1 - svf)) / (beam.sum() + diff.sum())
    return loss, svf


def azal_table(H):
    """PVWatts shading_azal: first row azimuths, first column altitudes, values % of cells shaded."""
    Hx = np.concatenate([H, H[:, :1]], axis=1)  # 360 = 0
    vals = (Hx[None, :, :] > AZAL_ALT[:, None, None]).mean(axis=1) * 100
    tab = np.zeros((len(AZAL_ALT) + 1, len(AZAL_AZ) + 1))
    tab[0, 1:], tab[1:, 0], tab[1:, 1:] = AZAL_AZ, AZAL_ALT, np.round(vals, 1)
    return tab.tolist()


def sample(idx_raster, step):
    """Cells of a 0.5 m id raster sampled every `step` cells -> rows, cols, ids, x, y."""
    r, c = np.nonzero(idx_raster[::step, ::step])
    r, c = r * step, c * step
    return r, c, idx_raster[r, c], X0 + (c + 0.5) * RES, Y1 - (r + 0.5) * RES


def apply_archetype(df):
    """Fill SAM design + mount / structure / cost fields from config archetypes (in place)."""
    A = CFG["archetypes"]
    get = lambda k: df["archetype"].map(lambda a: A[a][k])
    flush = df["archetype"].eq("roof_flush_pitch")
    df["array_type"] = get("array_type")
    df["tilt"] = np.where(flush, df.get("seg_tilt", pd.Series(np.nan, index=df.index)).round(0),
                          pd.to_numeric(get("tilt"), errors="coerce"))
    df["azimuth"] = np.where(flush, df.get("seg_azimuth", pd.Series(np.nan, index=df.index)).round(0),
                             df["row_azimuth_edge"])
    for k in ("gcr", "kw_per_m2", "mount", "structure", "cost_key"):
        df[k] = get(k)


def add_costs(df):
    """Placeholder capex from config (Felipe replaces the $/W); kept separate from the energy inputs."""
    usd = df["cost_key"].map(CFG["costs"]["installed_usd_per_wdc"])
    df["usd_per_wdc_placeholder"] = usd
    df["capex_musd_placeholder"] = (df["system_capacity_kw"] * usd / 1e3).round(2)


def dist_to_substation(gdf):
    """Distance from each site to the nearest TAMU electrical substation (interconnection proxy)."""
    subs = gpd.read_file(ROOT / "data/raw/tamu_fcor.gpkg", layer="electrical_substation").to_crs(gdf.crs)
    return gdf.geometry.distance(subs.union_all()).round(0)


# ---------------------------------------------------------------------------------------------
def roofs_part(obs, sun, loss_img):
    roofs = gpd.read_file(STEP2, layer="roofs")
    secs = pd.read_csv(V["interim_dir"] / "roof_sections_step2.csv")
    idx = load("usable_roof_section_idx")
    dsm_l = load("dsm_bldg" if (LID / "dsm_bldg.tif").exists() else "dsm_last")
    r, c, sid, x, y = sample(idx, 2)  # 1 m sample
    z0 = dsm_l[r, c] + PANEL_H_ROOF
    print(f"roof sample points: {len(x):,}")
    H = horizons(x, y, z0, obs)

    # design per section from the config archetypes:
    #   garage top decks stay in use -> elevated canopy; flat roofs -> ballasted rack;
    #   usable pitched faces -> flush mount at the face's own tilt / azimuth
    secs = secs.merge(roofs[["roof_id", "row_azimuth_edge", "tamu_type"]], on="roof_id", how="left")
    secs["archetype"] = np.select([secs["tamu_type"].eq("GARAGE"), secs["kind"].eq("flat")],
                                  ["garage_top_canopy", "roof_flat_rack"], "roof_flush_pitch")
    apply_archetype(secs)

    loss = np.zeros(len(x)); svf = np.zeros(len(x))
    cell_sec = sid - 1  # row in secs
    # group sections by orientation (tilt to 1 deg, azimuth to 5 deg) to share the TMY weights
    okey = (secs["tilt"].round().astype(int).astype(str) + "/" +
            (secs["azimuth"] / 5).round().mul(5).astype(int).astype(str))
    for key in okey.unique():
        rows_ = np.nonzero(okey.values == key)[0]
        m = np.isin(cell_sec, rows_)
        if m.any():
            t, a = map(float, key.split("/"))
            loss[m], svf[m] = shade_loss(H[m], t, a, sun)
    loss_img[r // 2, c // 2] = loss

    out = []
    for i in range(len(secs)):
        m = cell_sec == i
        n = m.sum()
        keep = m & (loss <= MAX_LOSS)
        pass_frac = keep.sum() / n if n else 0.0
        rec = dict(n_sample=int(n), shade_pass_frac=round(pass_frac, 3),
                   shade_loss_all_pct=round(100 * loss[m].mean(), 1) if n else np.nan,
                   shade_loss_pct=round(100 * loss[keep].mean(), 1) if keep.any() else np.nan,
                   shading_diff=round(100 * (1 - svf[keep].mean()), 1) if keep.any() else np.nan,
                   shading_azal=json.dumps(azal_table(H[keep])) if keep.any() else None)
        out.append(rec)
    secs = pd.concat([secs, pd.DataFrame(out)], axis=1)
    secs["final_plan_m2"] = (secs["usable_plan_m2"] * secs["shade_pass_frac"]).round(1)
    secs["final_surface_m2"] = (secs["usable_surface_m2"] * secs["shade_pass_frac"]).round(1)
    # racks and canopies are sized per m2 of plan area; flush modules per m2 of the sloped face
    flush = secs["archetype"].eq("roof_flush_pitch")
    secs["system_capacity_kw"] = np.where(flush, secs["final_surface_m2"], secs["final_plan_m2"]) * secs["kw_per_m2"]
    secs["system_capacity_kw"] = secs["system_capacity_kw"].round(1)
    secs = secs[secs["final_plan_m2"] >= 10].reset_index(drop=True)
    # roofs the 2017 LiDAR does not show (built later): its "sections" describe the old site,
    # so drop them; each such roof gets one estimated flat-rack section below
    stale = set(roofs.loc[~roofs["lidar_ok"], "roof_id"])
    secs = secs[~secs["roof_id"].isin(stale)].reset_index(drop=True)

    # roofs: LiDAR-measured totals + estimates for roofs the 2017 LiDAR does not see
    agg = secs.groupby("roof_id").agg(final_usable_m2=("final_plan_m2", "sum"),
                                      system_capacity_kw=("system_capacity_kw", "sum"),
                                      n_sections=("section_id", "count"))
    sh = secs.assign(w=secs.final_plan_m2 * secs.shade_loss_pct).groupby("roof_id")
    agg["shade_loss_pct"] = (sh.w.sum() / sh.final_plan_m2.sum()).round(1)
    roofs = roofs.drop(columns=[c for c in agg.columns if c in roofs], errors="ignore").merge(
        agg, left_on="roof_id", right_index=True, how="left")
    measured = roofs["lidar_ok"]
    shade_pass_med = (roofs.loc[measured, "final_usable_m2"] / roofs.loc[measured, "usable_lidar_m2"]).median()
    est = ~measured
    roofs.loc[est, "final_usable_m2"] = (roofs.loc[est, "usable_m2"] * shade_pass_med).round(1)
    est_arch = np.where(roofs["tamu_type"].eq("GARAGE"), "garage_top_canopy", "roof_flat_rack")
    est_kw_m2 = pd.Series(est_arch, index=roofs.index).map(lambda a: CFG["archetypes"][a]["kw_per_m2"])
    roofs.loc[est, "system_capacity_kw"] = (roofs.loc[est, "final_usable_m2"] * est_kw_m2[est]).round(1)
    roofs.loc[est, ["shade_loss_pct", "n_sections"]] = [np.nan, 1]
    roofs[["final_usable_m2", "system_capacity_kw", "n_sections"]] = \
        roofs[["final_usable_m2", "system_capacity_kw", "n_sections"]].fillna(0)

    # one placeholder section per estimated roof so every site has PySAM inputs
    e = roofs[est]
    est_secs = pd.DataFrame({
        "roof_id": e["roof_id"].values, "section_id": e["roof_id"].values + "-E01", "kind": "flat (estimated)",
        "archetype": est_arch[est.values], "row_azimuth_edge": e["row_azimuth_edge"].values,
        "final_plan_m2": e["final_usable_m2"].values, "final_surface_m2": e["final_usable_m2"].values,
        "system_capacity_kw": e["system_capacity_kw"].values,
        "shading_azal": None, "shading_diff": np.nan, "shade_loss_pct": np.nan,
    })
    apply_archetype(est_secs)
    secs = pd.concat([secs, est_secs], ignore_index=True)

    # roof-level design + cost drivers (main archetype = the one carrying most kW)
    main_arch = (secs.groupby(["roof_id", "archetype"])["system_capacity_kw"].sum()
                 .reset_index().sort_values("system_capacity_kw").groupby("roof_id").last()["archetype"])
    roofs["archetype"] = roofs["roof_id"].map(main_arch)
    for k in ("mount", "structure", "cost_key"):
        roofs[k] = roofs["archetype"].map(lambda a: CFG["archetypes"][a][k] if isinstance(a, str) else None)
    roofs["roof_age_yr"] = CURRENT_YEAR - roofs["year_built"]  # building age; re-roof dates unknown
    add_costs(roofs)
    return roofs, secs


def open_part(obs, sun, loss_img):
    o = gpd.read_file(STEP2, layer="open_land")
    idx = load("usable_open_id")
    dtm = load("dtm")
    r, c, oid, x, y = sample(idx, 4)  # 2 m sample
    z0 = dtm[r, c] + PANEL_H_GROUND
    print(f"open-land sample points: {len(x):,}")
    H = horizons(x, y, z0, obs)
    gf, gt = CFG["archetypes"]["ground_fixed"], CFG["archetypes"]["ground_tracker"]
    loss, svf = shade_loss(H, gf["tilt"], gf["azimuth"], sun)
    for dr in (0, 1):
        for dc in (0, 1):
            loss_img[np.minimum(r // 2 + dr, loss_img.shape[0] - 1), np.minimum(c // 2 + dc, loss_img.shape[1] - 1)] = loss
    out = []
    for i in range(len(o)):
        m = oid == i + 1
        keep = m & (loss <= MAX_LOSS)
        out.append(dict(shade_pass_frac=round(keep.sum() / m.sum(), 3) if m.any() else 0.0,
                        shade_loss_pct=round(100 * loss[keep].mean(), 1) if keep.any() else np.nan,
                        shading_diff=round(100 * (1 - svf[keep].mean()), 1) if keep.any() else np.nan,
                        shading_azal=json.dumps(azal_table(H[keep])) if keep.any() else None))
    o = pd.concat([o, pd.DataFrame(out)], axis=1)
    o["final_usable_m2"] = (o["usable_lidar_m2"] * o["shade_pass_frac"]).round(1)
    o["kw_fixed"] = (o["final_usable_m2"] * gf["kw_per_m2"]).round(1)
    o["kw_tracker"] = (o["final_usable_m2"] * gt["kw_per_m2"]).round(1)
    for kind, kw in (("ground_fixed", "kw_fixed"), ("ground_tracker", "kw_tracker")):
        usd = CFG["costs"]["installed_usd_per_wdc"][CFG["archetypes"][kind]["cost_key"]]
        o[f"capex_musd_{kind.split('_')[1]}_placeholder"] = (o[kw] * usd / 1e3).round(2)
    return o


def main():
    obs = obstacle_grid()
    sun = tmy()
    print(f"TMY daylight hours used: {len(sun)}")
    loss_img = np.full(obs.shape, np.nan, np.float32)

    roofs, secs = roofs_part(obs, sun, loss_img)
    o = open_part(obs, sun, loss_img)
    roofs["dist_substation_m"] = dist_to_substation(roofs)
    o["dist_substation_m"] = dist_to_substation(o)

    OUT.unlink(missing_ok=True)
    roofs.to_file(OUT, layer="roofs")
    o.to_file(OUT, layer="open_land")
    secs.to_csv(OUTD / "roof_sections.csv", index=False)
    roofs.drop(columns="geometry").to_csv(OUTD / "roofs.csv", index=False)
    o.drop(columns="geometry").to_csv(OUTD / "open_land.csv", index=False)
    with rasterio.open(LID / "shade_loss.tif", "w", driver="GTiff", height=obs.shape[0], width=obs.shape[1],
                       count=1, dtype="float32", crs="EPSG:6343", nodata=np.nan, compress="deflate",
                       transform=rasterio.transform.from_origin(X0, Y1, 1.0, 1.0)) as dst:
        dst.write(loss_img, 1)

    print(f"\n[{V['name']} LiDAR] ROOFS: usable after shading {roofs.final_usable_m2.sum():,.0f} m2, "
          f"{roofs.system_capacity_kw.sum() / 1e3:.1f} MWdc "
          f"({roofs.loc[~roofs.lidar_ok, 'system_capacity_kw'].sum() / 1e3:.1f} MWdc estimated, no LiDAR roof)")
    lid = secs[secs["shading_azal"].notna()]
    print(f"sections: {len(secs)} ({len(lid)} LiDAR-measured); weighted shade loss on kept cells "
          f"{np.average(lid.shade_loss_pct, weights=lid.final_plan_m2):.1f} %; "
          f"share of LiDAR-usable area kept after shading {lid.final_plan_m2.sum() / lid.usable_plan_m2.sum():.0%}")
    print(f"OPEN: {o.final_usable_m2.sum() / 1e4:.1f} ha after shading -> "
          f"{o.kw_fixed.sum() / 1e3:.1f} MWdc fixed or {o.kw_tracker.sum() / 1e3:.1f} MWdc tracker")


if __name__ == "__main__":
    main()
