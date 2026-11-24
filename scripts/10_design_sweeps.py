"""Step 8 (proposal O2): design sweeps per land type with PVWatts v8, inside a practical envelope.

  python 10_design_sweeps.py            (archetype level: 1 MWdc, unshaded, College Station TMY)

Rooftop   : tilt 0-20 deg x GCR 0.4-0.95, south-row azimuth (campus grid 140 / 230 deg vs 180),
            plus east-west dual-tilt racks.                   limit tested: LAND (kWh per m2 of roof)
Carport   : tilt 0-15 deg, azimuth set by the stalls (campus grid) vs south.  limit: BUDGET
Open land : fixed tilt 10-35 deg x GCR 0.3-0.7; 1-axis tracker (backtracking) GCR 0.25-0.6;
            DC/AC 1.0-1.6 at a fixed AC size.            limits: LAND, BUDGET, GRID (clipping)

PVWatts models row-to-row self-shading from GCR (fixed racks) and backtracking (trackers).
Land density: kWdc per m2 = module kW/m2 (config, incl. inter-module gaps) x GCR.

Feasibility envelope (what can actually be built - industry practice, flagged per case):
  roof ballasted rack : tilt 5-15 deg (>= 5 self-cleaning / no ponding; <= 15 wind uplift / ballast),
                        clear row gap >= 0.3 m
  roof east-west rack : tilt 5-15 deg, GCR <= 0.95 (service aisles)
  carport             : tilt 3-10 deg (drainage >= 3, clearance / wind <= 10); azimuth fixed by the stalls
                        (campus grid 140 / 230 deg), so "due south" is generally not available
  ground fixed        : clear row gap >= 1.0 m (mowing, O&M access)
  tracker             : GCR 0.25-0.50, rotation +-45 deg
  DC/AC               : 1.1-1.5 (inverter warranty / clipping practice)
Outputs: outputs/design_sweeps/*.csv
"""
import json

import numpy as np
import pandas as pd
import PySAM.Pvwattsv8 as pv

from vintage import ROOT

CFG = json.loads((ROOT / "config/sam_defaults.json").read_text())
OUT = ROOT / "outputs/design_sweeps"
WEATHER = str(ROOT / CFG["weather_file"])
KW_M2 = CFG["module"]["kw_per_m2_module"]
USD = CFG["costs"]["installed_usd_per_wdc"]
# module (Hyperion HY-WH108P8-410, 1.72 x 1.13 m): roof rows 1 module landscape, ground 2 in portrait
ROW_DEPTH = {"roof": 1.13, "ground": 2 * 1.72 + 0.02}
NO_ROWS_GCR = 0.1            # PVWatts GCR for a continuous plane (canopy) / no-row reference
CAMPUS_GRID_AZ = (140, 230)  # edge-aligned rows on the ~50 deg campus grid (most roofs, lots)


def pvwatts(array_type, tilt, azimuth, gcr, dc_ac=1.2, kwdc=1000.0):
    m = pv.default("PVWattsNone")
    m.SolarResource.solar_resource_file = WEATHER
    m.SolarResource.use_wf_albedo = 0
    m.SolarResource.albedo = [CFG["albedo"]] * 12
    d = m.SystemDesign
    d.system_capacity, d.array_type = float(kwdc), float(array_type)
    d.tilt, d.azimuth, d.gcr, d.dc_ac_ratio = float(tilt), float(azimuth), float(gcr), float(dc_ac)
    d.inv_eff, d.losses, d.module_type = float(CFG["inv_eff"]), float(CFG["losses_pct_site_shading"]), 1.0
    if array_type in (2, 3):
        d.rotlim = 45.0
    m.execute(0)
    ac = np.asarray(m.Outputs.ac) / 1e3        # kW
    dc = np.asarray(m.Outputs.dc) / 1e3        # kW
    ac_rated = kwdc / dc_ac
    pot = dc * CFG["inv_eff"] / 100            # AC before the inverter limit
    clip = np.clip(pot - ac_rated, 0, None).sum() / max(pot.sum(), 1)
    return dict(kwh_per_kwdc=ac.sum() / kwdc, clipping_pct=100 * clip,
                mwh_per_mwac=ac.sum() / ac_rated, cf_ac_pct=100 * ac.sum() / (ac_rated * 8760))


def roof_rows(tilt, azimuth, gcr, dc_ac=1.2, kwdc=1000.0):
    """Low-tilt rows on a roof. PVWatts 'roof mount' (array_type 1) has the right cell temperature but no
    row-to-row shading; the shading derate is taken from open-rack runs at the same tilt / azimuth / GCR."""
    r = pvwatts(1, tilt, azimuth, NO_ROWS_GCR, dc_ac, kwdc)
    f = pvwatts(0, tilt, azimuth, gcr)["kwh_per_kwdc"] / pvwatts(0, tilt, azimuth, NO_ROWS_GCR)["kwh_per_kwdc"]
    return {**r, "kwh_per_kwdc": r["kwh_per_kwdc"] * f, "mwh_per_mwac": r["mwh_per_mwac"] * f,
            "cf_ac_pct": r["cf_ac_pct"] * f, "self_shading_pct": 100 * (1 - f)}


def row_gap(gcr, tilt, depth):
    """Clear gap between rows [m] for module depth `depth` (slant) at `tilt`."""
    pitch = depth / gcr
    return pitch - depth * np.cos(np.radians(tilt))


def gcr_dec21(tilt):
    """GCR from the team's no-shade rule on Dec 21, 9-15 h: d = 1.89 h (row spacing / height)."""
    t = np.radians(tilt)
    return 1 / (np.cos(t) + 1.89 * np.sin(t))


def add_land(df):
    df["kwdc_per_m2"] = KW_M2 * df["gcr"]
    df["kwh_per_m2"] = df["kwh_per_kwdc"] * df["kwdc_per_m2"]
    return df


def roof_sweep():
    rows = []
    for tilt in [0, 5, 10, 15, 20]:
        for gcr in [0.4, 0.5, 0.6, 0.7, 0.76, 0.8, 0.85, 0.9, 0.95]:
            for az in (180, *CAMPUS_GRID_AZ):
                r = roof_rows(tilt, az, gcr)
                gap = row_gap(gcr, tilt, ROW_DEPTH["roof"])
                feas = 5 <= tilt <= 15 and gap >= 0.3
                rows.append(dict(layout="south rows" if az == 180 else f"edge-aligned {az}", tilt=tilt, azimuth=az,
                                 gcr=gcr, row_gap_m=round(gap, 2), feasible=feas, **r))
        if tilt > 0:  # east-west dual tilt: half the modules face 90, half 270, back to back
            for gcr in [0.8, 0.9, 0.95]:
                e, w = roof_rows(tilt, 90, gcr), roof_rows(tilt, 270, gcr)
                r = {k: (e[k] + w[k]) / 2 for k in e}
                rows.append(dict(layout="east-west dual tilt", tilt=tilt, azimuth=90, gcr=gcr, row_gap_m=0.0,
                                 feasible=5 <= tilt <= 15, **r))
    df = add_land(pd.DataFrame(rows))
    df["dec21_gcr"] = df["tilt"].map(lambda t: round(gcr_dec21(t), 3) if t > 0 else 1.0)
    return df


def carport_sweep():
    rows = []
    for tilt in [0, 3, 5, 7, 10, 15]:
        for az in (180, *CAMPUS_GRID_AZ, 320, 50):  # 320 / 50 = canopy facing away from south (worst case)
            r = pvwatts(0, tilt, az, NO_ROWS_GCR)  # one continuous canopy per stall row: no row shading
            rows.append(dict(tilt=tilt, azimuth=az, gcr=0.95, feasible=(3 <= tilt <= 10) and az in CAMPUS_GRID_AZ, **r))
    return add_land(pd.DataFrame(rows))


def ground_sweep():
    rows = []
    for tilt in [10, 15, 20, 25, 30, 35]:
        for gcr in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
            r = pvwatts(0, tilt, 180, gcr)
            gap = row_gap(gcr, tilt, ROW_DEPTH["ground"])
            rows.append(dict(mount="fixed", tilt=tilt, gcr=gcr, row_gap_m=round(gap, 2), feasible=gap >= 1.0,
                             usd_per_wdc=USD["ground_fixed"], **r))
    for gcr in [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]:
        r = pvwatts(3, 0, 180, gcr)
        rows.append(dict(mount="tracker", tilt=0, gcr=gcr, row_gap_m=np.nan, feasible=0.25 <= gcr <= 0.5,
                         usd_per_wdc=USD["ground_tracker"], **r))
    df = add_land(pd.DataFrame(rows))
    df["kwh_per_usd"] = df["kwh_per_kwdc"] / (1000 * df["usd_per_wdc"])  # first-year kWh per $ capex
    return df


def dcac_sweep():
    """Fixed AC size (grid / inverter limit): add DC per MWac and watch clipping."""
    rows = []
    for mount, at, tilt, gcr in [("fixed 25 deg", 0, 25, 0.5), ("tracker", 3, 0, 0.35), ("roof 10 deg", 1, 10, 0.76)]:
        for dcac in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]:
            az = 180 if mount != "roof 10 deg" else 140
            r = (roof_rows(tilt, az, gcr, dc_ac=dcac, kwdc=1000 * dcac) if mount == "roof 10 deg"
                 else pvwatts(at, tilt, az, gcr, dc_ac=dcac, kwdc=1000 * dcac))
            rows.append(dict(mount=mount, dc_ac=dcac, mwdc_per_mwac=dcac, feasible=1.1 <= dcac <= 1.5, **r))
    return pd.DataFrame(rows)


def best(df, metric, label, extra=None, where=None):
    f = df[df.feasible] if where is None else df[df.feasible & where]
    b = f.loc[f[metric].idxmax()]
    cols = [c for c in ["layout", "mount", "tilt", "azimuth", "gcr", "dc_ac", "kwh_per_kwdc", "kwdc_per_m2",
                        "kwh_per_m2", "clipping_pct", "kwh_per_usd"] if c in df]
    return {"case": label, "metric": metric, **{c: b[c] for c in cols}, **(extra or {})}


def inventory_effect(roof, carport, ground, vintage="2025"):
    """What each design would deliver on the inventory's usable areas (site shading kept as is)."""
    d = ROOT / f"outputs/{vintage}"
    rs, ps, ol = (pd.read_csv(d / f) for f in ("roof_sections.csv", "parking_sections.csv", "open_land.csv"))
    flat = rs[rs.archetype == "roof_flat_rack"]
    roof_m2 = flat.final_plan_m2.sum()
    roof_shade = 1 - np.average(flat.shade_loss_pct.fillna(flat.shade_loss_pct.median()), weights=flat.final_plan_m2) / 100
    car_m2, car_shade = ps.final_canopy_m2.sum(), 1 - np.average(ps.shade_loss_pct, weights=ps.final_canopy_m2) / 100
    land_m2, land_shade = ol.final_usable_m2.sum(), 1 - np.average(ol.shade_loss_pct.fillna(0), weights=ol.final_usable_m2.clip(lower=1e-9)) / 100
    pick = lambda df, **kw: df[np.logical_and.reduce([df[k] == v for k, v in kw.items()])].iloc[0]
    cases = [
        ("flat roofs", "current: 10 deg rows, 140, GCR 0.76", pick(roof, layout="edge-aligned 140", tilt=10, gcr=0.76), roof_m2, roof_shade),
        ("flat roofs", "5 deg rows, 140, Dec-21 GCR 0.85", pick(roof, layout="edge-aligned 140", tilt=5, gcr=0.85), roof_m2, roof_shade),
        ("flat roofs", "east-west 10 deg, GCR 0.95", pick(roof, layout="east-west dual tilt", tilt=10, gcr=0.95), roof_m2, roof_shade),
        ("flat roofs", "east-west 5 deg, GCR 0.95", pick(roof, layout="east-west dual tilt", tilt=5, gcr=0.95), roof_m2, roof_shade),
        ("carports", "current: 7 deg, stall-aligned 140", None, car_m2, car_shade),
        ("carports", "5 deg, stall-aligned 140", pick(carport, tilt=5, azimuth=140), car_m2, car_shade),
        ("open land", "fixed 25 deg, GCR 0.5 (current)", pick(ground, mount="fixed", tilt=25, gcr=0.5), land_m2, land_shade),
        ("open land", "fixed 20 deg, GCR 0.7 (max per m2)", pick(ground, mount="fixed", tilt=20, gcr=0.7), land_m2, land_shade),
        ("open land", "tracker GCR 0.35 (current)", pick(ground, mount="tracker", gcr=0.35), land_m2, land_shade),
        ("open land", "tracker GCR 0.5 (max per m2)", pick(ground, mount="tracker", gcr=0.5), land_m2, land_shade),
    ]
    rows = []
    for kind, label, r, m2, shade in cases:
        if r is None:
            r = pick(carport, tilt=7, azimuth=140)
        if kind == "carports":  # canopy density is set by the canopy (0.18 kW/m2), not a row GCR
            r = r.copy(); r["kwdc_per_m2"] = CFG["archetypes"]["carport"]["kw_per_m2"]; r["kwh_per_m2"] = r.kwh_per_kwdc * r.kwdc_per_m2
        rows.append(dict(land=kind, design=label, area_ha=round(m2 / 1e4, 1), MWdc=round(m2 * r.kwdc_per_m2 / 1e3, 1),
                         GWh=round(m2 * r.kwh_per_m2 * shade / 1e6, 1), kwh_per_kwdc=round(r.kwh_per_kwdc)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"inventory_effect_{vintage}.csv", index=False)
    print("\nOn the 2025 inventory areas (site shading kept):")
    print(df.to_string(index=False))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    roof, carport, ground, dcac = roof_sweep(), carport_sweep(), ground_sweep(), dcac_sweep()
    for name, df in [("roof", roof), ("carport", carport), ("ground", ground), ("dcac", dcac)]:
        df.round(4).to_csv(OUT / f"sweep_{name}.csv", index=False)

    picks = [
        best(roof[roof.layout != "east-west dual tilt"], "kwh_per_m2", "roof, rows: max kWh per m2 of roof"),
        best(roof[roof.layout == "east-west dual tilt"], "kwh_per_m2", "roof, east-west: max kWh per m2 of roof"),
        best(roof, "kwh_per_kwdc", "roof: max kWh per module (kWdc)"),
        best(carport, "kwh_per_kwdc", "carport: max kWh per kWdc"),
        best(ground[ground.mount == "fixed"], "kwh_per_m2", "ground fixed: max kWh per m2 of land"),
        best(ground[ground.mount == "tracker"], "kwh_per_m2", "tracker: max kWh per m2 of land"),
        best(ground, "kwh_per_usd", "ground: max kWh per $ (budget)"),
        best(ground, "kwh_per_kwdc", "ground: max kWh per kWdc"),
    ] + [best(dcac[dcac["mount"] == m], "mwh_per_mwac", f"{m}: max MWh per MWac (grid limit, clipping <= 3 %)",
              where=dcac["clipping_pct"] <= 3) for m in dcac["mount"].unique()]
    picks += [best(roof[(roof.layout == "edge-aligned 140") & (roof.gcr == 0.76)], "kwh_per_m2",
                   "CURRENT roof archetype (10 deg, 140, GCR 0.76)", where=roof.tilt == 10)]
    pd.DataFrame(picks).round(3).to_csv(OUT / "best_designs.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(picks).round(3).to_string(index=False))
    inventory_effect(roof, carport, ground)
    print("\nDC/AC at fixed AC size:")
    print(dcac.pivot(index="dc_ac", columns="mount", values=["mwh_per_mwac", "clipping_pct"]).round(1).to_string())


if __name__ == "__main__":
    main()
