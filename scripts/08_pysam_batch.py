"""Step 6: PySAM (PVWatts v8) run for every section of the inventory.

  python 08_pysam_batch.py --vintage 2017|2025

Roof racks: PVWatts 'roof mount' thermal model x row-shading derate from open-rack runs (roof mount has
no self-shading); carport / garage canopies are continuous planes (no row shading).
One PVWatts system per row of roof_sections.csv, parking_sections.csv and per open-land tract
(fixed 25 deg and 1-axis tracker, run separately), each with its own LiDAR shading tables.
Settings from config/sam_defaults.json; College Station TMY; first-year energy (no degradation).

Outputs (outputs/<v>/):
  energy_sections.csv      annual kWh, specific yield, capacity factor per section / tract
  energy_hourly_mw.csv     8760 h AC output [MW] summed by land type (for load matching)
"""
import json

import numpy as np
import pandas as pd
import PySAM.Pvwattsv8 as pv

from vintage import ROOT, get_vintage

V = get_vintage()
CFG = json.loads((ROOT / "config/sam_defaults.json").read_text())
OUTD = V["out_dir"]
WEATHER = str(ROOT / CFG["weather_file"])
PVWATTS_ARRAY = {0: 0, 1: 1, 2: 3}  # config array_type -> PVWatts (tracker runs with backtracking = 3)


def run(capacity_kw, array_type, tilt, azimuth, gcr, azal=None, diff=None):
    m = pv.default("PVWattsNone")
    m.SolarResource.solar_resource_file = WEATHER
    m.SolarResource.use_wf_albedo = 0
    m.SolarResource.albedo = [CFG["albedo"]] * 12
    d = m.SystemDesign
    d.system_capacity = float(capacity_kw)
    d.array_type = float(PVWATTS_ARRAY[int(array_type)])
    d.tilt, d.azimuth, d.gcr = float(tilt), float(azimuth), float(gcr)
    d.dc_ac_ratio, d.inv_eff = CFG["dc_ac_ratio"], CFG["inv_eff"]
    d.losses, d.module_type = CFG["losses_pct_site_shading"], CFG["module"]["module_type"]
    if azal is not None:
        m.Shading.shading_en_azal = 1
        m.Shading.shading_azal = json.loads(azal)
    if diff is not None and np.isfinite(diff):
        m.Shading.shading_en_diff = 1
        m.Shading.shading_diff = float(diff)
    m.execute()
    return np.asarray(m.Outputs.ac) / 1e3  # kW (8760)


NO_ROWS_GCR = 0.1
_rsf = {}


def row_shading_factor(tilt, azimuth, gcr):
    """Row-to-row shading derate for roof racks: PVWatts 'roof mount' has no self-shading, so take the
    ratio from open-rack runs (with / without rows) at the same tilt, azimuth and GCR (cached)."""
    key = (round(float(tilt)), round(float(azimuth) / 5) * 5, round(float(gcr), 2))
    if key not in _rsf:
        _rsf[key] = run(1000, 0, key[0], key[1], key[2]).sum() / run(1000, 0, key[0], key[1], NO_ROWS_GCR).sum()
    return _rsf[key]


def main():
    jobs = []
    roof = pd.read_csv(OUTD / "roof_sections.csv")
    for r in roof.itertuples():
        kind = "garage canopy" if r.archetype == "garage_top_canopy" else "roof"
        jobs.append((kind, r.section_id, r.roof_id, r.system_capacity_kw, r.array_type, r.tilt, r.azimuth, r.gcr,
                     r.shading_azal if isinstance(r.shading_azal, str) else None, r.shading_diff))
    park = pd.read_csv(OUTD / "parking_sections.csv")
    for r in park.itertuples():
        jobs.append(("parking carport", r.section_id, r.lot_id, r.system_capacity_kw, r.array_type, r.tilt,
                     r.azimuth, r.gcr, r.shading_azal, r.shading_diff))
    # estimated lots without stall lines (no sections): unshaded carport archetype
    lots = pd.read_csv(OUTD / "parking_lots.csv")
    a = CFG["archetypes"]["carport"]
    for r in lots[lots.canopy_source.fillna("").str.startswith("estimate")].itertuples():
        jobs.append(("parking carport", r.lot_id + "-E", r.lot_id, r.system_capacity_kw, a["array_type"], a["tilt"],
                     180, a["gcr"], None, None))
    op = pd.read_csv(OUTD / "open_land.csv")
    for kind, kw_col in (("ground_fixed", "kw_fixed"), ("ground_tracker", "kw_tracker")):
        a = CFG["archetypes"][kind]
        for r in op[op[kw_col] > 0].itertuples():
            jobs.append((f"open land ({kind.split('_')[1]})", f"{r.open_id}-{kind.split('_')[1]}", r.open_id,
                         getattr(r, kw_col), a["array_type"], a["tilt"], a["azimuth"], a["gcr"],
                         r.shading_azal if isinstance(r.shading_azal, str) else None, r.shading_diff))

    rows, hourly = [], {}
    for kind, sid, site, kw, at, tilt, az, gcr, azal, diff in jobs:
        if not kw or kw <= 0:
            continue
        if kind in ("garage canopy", "parking carport"):
            ac = run(kw, at, tilt, az, NO_ROWS_GCR, azal, diff)         # continuous canopy, no rows
        elif kind == "roof" and gcr < 0.9:                                # rack rows (flush faces use gcr 0.9)
            ac = run(kw, at, tilt, az, NO_ROWS_GCR, azal, diff) * row_shading_factor(tilt, az, gcr)
        else:
            ac = run(kw, at, tilt, az, gcr, azal, diff)
        e = ac.sum()
        rows.append(dict(kind=kind, section_id=sid, site_id=site, kwdc=round(kw, 1), annual_kwh=round(e),
                         kwh_per_kwdc=round(e / kw, 1), cf_dc_pct=round(100 * e / (kw * 8760), 2),
                         peak_kwac=round(ac.max(), 1)))
        hourly[kind] = hourly.get(kind, 0) + ac / 1e3
    res = pd.DataFrame(rows)
    res.to_csv(OUTD / "energy_sections.csv", index=False)
    pd.DataFrame(hourly).round(4).to_csv(OUTD / "energy_hourly_mw.csv", index_label="hour")

    s = res.groupby("kind").agg(sections=("section_id", "count"), MWdc=("kwdc", lambda x: x.sum() / 1e3),
                                GWh=("annual_kwh", lambda x: x.sum() / 1e6))
    s["kWh/kWdc"] = (1e3 * s.GWh / s.MWdc).round(0)
    print(f"[{V['name']}] PVWatts runs: {len(res)}")
    print(s.round(2).to_string())


if __name__ == "__main__":
    main()
