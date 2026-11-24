"""Step 9 (proposal O3/O4): portfolios from the site inventory, scored on land, cost and the grid.

  python 11_portfolios.py --vintage 2025

Site options (one design chosen per site), energy = PySAM per site with its own LiDAR shading
(08_pysam_batch); alternative designs scaled by the archetype sweeps (10_design_sweeps):
  flat roofs   : rows 10 deg / campus grid / GCR 0.76 (base)  |  east-west 5 deg, GCR 0.95
  pitched roof : flush at the face's tilt / azimuth
  garage tops  : steel canopy 7 deg                           |  surface lots: carport canopy 7 deg
  open land    : fixed 25 deg GCR 0.5 (base) | fixed 20 deg GCR 0.6 (dense) |
                 tracker GCR 0.35 (base)     | tracker GCR 0.5 (dense)

Portfolios (targets: 5 % and 10 % of FY2026 campus use, central 650 GWh):
  A  least land   : roofs, then garages, then parking (no new land), open land last; land-efficient designs
  B  least cost   : cheapest kWh first (lowest LCOE option per site), any land
  B0 least cost, built land only (no open land)
  C  maximum output: every site with its highest-energy design (C0: built land only)
  D  grid-limited : cheapest kWh first over all sites (any land type), a site is skipped if adding it would
                    push export to ERCOT above 5 % of PV output -> the most PV the campus can absorb
Each portfolio: MWdc, GWh, % campus, new land, capex, LCOE, payback; and hour-by-hour vs the campus
load minus CHP (09_campus_balance, central, cooling shape): used on campus, exported, peak MWac.
Costs are the placeholders in config/sam_defaults.json -> costs.

Outputs: outputs/<v>/portfolios/summary.csv, sites_<portfolio>.csv, site_options.csv, open_land_policy.csv
"""
import importlib.util
import json
import re

import numpy as np
import pandas as pd
import PySAM.Pvwattsv8 as pv

from vintage import ROOT, get_vintage

V = get_vintage()
OUTD = V["out_dir"]
PD = OUTD / "portfolios"
CFG = json.loads((ROOT / "config/sam_defaults.json").read_text())
COST, FIN = CFG["costs"], CFG["costs"]["finance"]
SW = ROOT / "outputs/design_sweeps"
spec = importlib.util.spec_from_file_location("cb", ROOT / "scripts/09_campus_balance.py")
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)

CAMPUS = cb.CAMPUS_GWH["central"]
TARGETS = {"5%": 0.05 * CAMPUS, "10%": 0.10 * CAMPUS}
NO_ROWS = 0.1
POLO = "O0001"  # Polo Fields tract (Bonfire Memorial cut out): in use as fields - shown with / without
# open-land policy screen (portfolio level): no parks, memorials, gardens, rec / sports fields, golf,
# research farmland, retail / construction sites (future buildings); 10 % overlap is enough to exclude
OPEN_EXCLUDE_OSM = {"pitch", "garden", "park", "golf_course", "farmland", "retail", "construction"}


def open_policy(ol):
    reason = pd.Series("", index=ol.index)
    lu = ol.landuse_name.fillna("")
    reason[(ol.landuse_frac >= 0.1) & ~lu.str.startswith("MEMRL")] = "TAMU land-use record (park / rec / garden)"
    reason[(ol.park_frac >= 0.1) & (reason == "")] = "campus park / quad"
    reason[(ol.golf_frac > 0) & (reason == "")] = "golf course"
    reason[(ol.sports_frac >= 0.1) & (reason == "")] = "sports field"
    reason[ol.osm_name.isin(OPEN_EXCLUDE_OSM) & (ol.osm_frac >= 0.1) & (reason == "")] = "OSM: " + ol.osm_name.astype(str)
    return reason


# ---------------------------------------------------------------- archetype shapes and ratios
def hourly(array_type, tilt, az, gcr):
    m = pv.default("PVWattsNone")
    m.SolarResource.solar_resource_file = str(ROOT / CFG["weather_file"])
    m.SolarResource.use_wf_albedo = 0
    m.SolarResource.albedo = [CFG["albedo"]] * 12
    d = m.SystemDesign
    d.system_capacity, d.array_type, d.tilt, d.azimuth, d.gcr = 1000.0, float(array_type), float(tilt), float(az), float(gcr)
    d.dc_ac_ratio, d.inv_eff, d.losses, d.module_type = CFG["dc_ac_ratio"], float(CFG["inv_eff"]), CFG["losses_pct_site_shading"], 1.0
    if array_type == 3:
        d.rotlim = 45.0
    m.execute(0)
    return np.asarray(m.Outputs.ac) / 1e3


def shapes():
    """Normalised 8760 profiles (sum = 1) per design."""
    s = {
        "roof rows 10": hourly(1, 10, 140, NO_ROWS),
        "roof east-west 5": (hourly(1, 5, 90, NO_ROWS) + hourly(1, 5, 270, NO_ROWS)) / 2,
        "roof flush": hourly(1, 20, 180, NO_ROWS),
        "canopy 7": hourly(0, 7, 140, NO_ROWS),
        "fixed 25/0.5": hourly(0, 25, 180, 0.5),
        "fixed 20/0.6": hourly(0, 20, 180, 0.6),
        "tracker 0.35": hourly(3, 0, 180, 0.35),
        "tracker 0.5": hourly(3, 0, 180, 0.5),
    }
    return {k: v / v.sum() for k, v in s.items()}


def sweep_ratio():
    r, g = pd.read_csv(SW / "sweep_roof.csv"), pd.read_csv(SW / "sweep_ground.csv")
    pick = lambda df, **kw: df[np.logical_and.reduce([np.isclose(df[k], v) if not isinstance(v, str) else df[k] == v
                                                      for k, v in kw.items()])].iloc[0]
    base_roof = pick(r, layout="edge-aligned 140", tilt=10, gcr=0.76)
    ew = pick(r, layout="east-west dual tilt", tilt=5, gcr=0.95)
    f25, f20 = pick(g, mount="fixed", tilt=25, gcr=0.5), pick(g, mount="fixed", tilt=20, gcr=0.6)
    t35, t50 = pick(g, mount="tracker", gcr=0.35), pick(g, mount="tracker", gcr=0.5)
    return {  # (kW multiplier, energy multiplier) relative to the base design on the same area
        "roof east-west 5": (ew.kwdc_per_m2 / base_roof.kwdc_per_m2, ew.kwh_per_m2 / base_roof.kwh_per_m2),
        "fixed 20/0.6": (f20.kwdc_per_m2 / f25.kwdc_per_m2, f20.kwh_per_m2 / f25.kwh_per_m2),
        "tracker 0.5": (t50.kwdc_per_m2 / t35.kwdc_per_m2, t50.kwh_per_m2 / t35.kwh_per_m2),
    }


# ---------------------------------------------------------------- site options
def site_options():
    e = pd.read_csv(OUTD / "energy_sections.csv")
    rs = pd.read_csv(OUTD / "roof_sections.csv")[["section_id", "archetype", "final_plan_m2"]]
    roofs = pd.read_csv(OUTD / "roofs.csv")
    lots = pd.read_csv(OUTD / "parking_lots.csv")
    ol = pd.read_csv(OUTD / "open_land.csv")
    ratio = sweep_ratio()
    opts = []

    def add(site_id, name, land_type, design, kwdc, kwh, area_m2, new_land_m2, cost_key, shape, lat=None, lon=None):
        capex = kwdc * 1e3 * COST["installed_usd_per_wdc"][cost_key]
        om = kwdc * COST["om_usd_per_kwdc_yr"][cost_key]
        lcoe = (capex * FIN["crf"] + om) / (kwh * FIN["levelized_energy_factor"])
        opts.append(dict(site_id=site_id, name=name, land_type=land_type, design=design, kwdc=kwdc, kwh=kwh,
                         area_m2=area_m2, new_land_m2=new_land_m2, capex_usd=capex, om_usd_yr=om,
                         lcoe_usd_kwh=lcoe, kwh_per_m2=kwh / max(area_m2, 1), shape=shape))

    # roofs (incl. garages): split by archetype
    r = e[e.kind.isin(["roof", "garage canopy"])].merge(rs, on="section_id", how="left")
    r["archetype"] = r["archetype"].fillna("roof_flat_rack")
    names = roofs.set_index("roof_id")["name"].fillna("(unnamed)")
    for rid, g in r.groupby("site_id"):
        nm = names.get(rid, rid)
        gar = g[g.archetype == "garage_top_canopy"]
        flat = g[g.archetype == "roof_flat_rack"]
        flush = g[g.archetype == "roof_flush_pitch"]
        area = g.final_plan_m2.fillna(0).sum()
        if len(gar):
            add(rid, nm, "garage top", "canopy 7", gar.kwdc.sum(), gar.annual_kwh.sum(), area, 0, "garage_canopy", "canopy 7")
            continue
        fk, fe, pk, pe = flat.kwdc.sum(), flat.annual_kwh.sum(), flush.kwdc.sum(), flush.annual_kwh.sum()
        if fk + pk <= 0:
            continue
        add(rid, nm, "roof", "rows 10", fk + pk, fe + pe, area, 0, "roof", "roof rows 10")
        if fk > 0:
            km, em = ratio["roof east-west 5"]
            add(rid, nm, "roof", "east-west 5", fk * km + pk, fe * em + pe, area, 0, "roof_east_west", "roof east-west 5")

    lot_name = lots.set_index("lot_id")["lot_name"]
    lot_area = lots.set_index("lot_id")["final_canopy_m2"]
    for lid, g in e[e.kind == "parking carport"].groupby("site_id"):
        add(lid, lot_name.get(lid, lid), "parking", "canopy 7", g.kwdc.sum(), g.annual_kwh.sum(),
            lot_area.get(lid, 0), 0, "carport", "canopy 7")

    ol["policy_excluded"] = open_policy(ol)
    ol[["open_id", "area_m2", "final_usable_m2", "landuse_name", "osm_name", "policy_excluded"]].to_csv(
        PD / "open_land_policy.csv", index=False)
    print(f"open land policy screen: {(ol.policy_excluded != '').sum()} of {len(ol)} tracts excluded "
          f"({ol.loc[ol.policy_excluded != '', 'final_usable_m2'].sum() / 1e4:.1f} of {ol.final_usable_m2.sum() / 1e4:.1f} ha)")
    keep = set(ol.loc[ol.policy_excluded == "", "open_id"])
    area = ol.set_index("open_id")["final_usable_m2"]
    nm = ol.set_index("open_id").apply(lambda x: x.get("landuse_name") if isinstance(x.get("landuse_name"), str)
                                       else (x.get("osm_name") if isinstance(x.get("osm_name"), str) else ""), axis=1)
    for kind, base, dense, ck in (("open land (fixed)", "fixed 25/0.5", "fixed 20/0.6", "ground_fixed"),
                                  ("open land (tracker)", "tracker 0.35", "tracker 0.5", "ground_tracker")):
        for oid, g in e[e.kind == kind].groupby("site_id"):
            if oid not in keep:
                continue
            a = area.get(oid, 0)
            label = "Polo Fields (Bonfire Memorial cut out)" if oid == POLO else f"{oid} {nm.get(oid, '')}".strip()
            add(oid, label, "open land", base, g.kwdc.sum(), g.annual_kwh.sum(), a, a, ck, base)
            km, em = ratio[dense]
            add(oid, label, "open land", dense, g.kwdc.sum() * km, g.annual_kwh.sum() * em, a, a, ck, dense)
    o = pd.DataFrame(opts)
    o = o[o.kwh > 0].reset_index(drop=True)
    return o


# ---------------------------------------------------------------- portfolio assembly
def fill(cands, target_kwh=None):
    """Take candidates in order until the target is met (whole sites; last site trimmed to fit)."""
    out, tot = [], 0.0
    for _, r in cands.iterrows():
        if target_kwh is not None and tot >= target_kwh:
            break
        r = r.copy()
        if target_kwh is not None and tot + r.kwh > target_kwh:  # partial last site (build part of it)
            f = (target_kwh - tot) / r.kwh
            for c in ("kwdc", "kwh", "area_m2", "new_land_m2", "capex_usd", "om_usd_yr"):
                r[c] *= f
            r["partial_frac"] = round(f, 3)
        out.append(r)
        tot += r.kwh
    return pd.DataFrame(out)


def one_per_site(o, key, ascending):
    return o.sort_values(key, ascending=ascending).drop_duplicates("site_id")


def build(o):
    order = {"roof": 0, "garage top": 1, "parking": 2, "open land": 3}
    land_eff = one_per_site(o, "kwh_per_m2", False)              # most energy per m2 per site
    land_eff = land_eff.assign(o=land_eff.land_type.map(order)).sort_values(["o", "kwh"], ascending=[True, False])
    cheap = one_per_site(o, "lcoe_usd_kwh", True).sort_values("lcoe_usd_kwh")
    maxe = one_per_site(o, "kwh", False)
    P = {}
    for t, kwh in TARGETS.items():
        P[f"A least land {t}"] = fill(land_eff, kwh * 1e6)
        P[f"B least cost {t}"] = fill(cheap, kwh * 1e6)
        P[f"B1 least cost, without Polo Fields {t}"] = fill(cheap[cheap.site_id != POLO], kwh * 1e6)
        P[f"B0 least cost, no open land {t}"] = fill(cheap[cheap.land_type != "open land"], kwh * 1e6)
    P["C maximum output (all sites)"] = maxe
    P["C0 maximum output, no open land"] = maxe[maxe.land_type != "open land"]
    return P


def grid_limited(o, S, load, max_export=0.05):
    cands = one_per_site(o, "lcoe_usd_kwh", True).sort_values("lcoe_usd_kwh")
    prof = np.zeros(8760)
    chosen = []
    for _, r in cands.iterrows():
        trial = prof + r["kwh"] * S[r["shape"]] / 1e3
        if np.maximum(trial - load, 0).sum() / trial.sum() > max_export:
            continue
        prof = trial
        chosen.append(r)
    return pd.DataFrame(chosen)


def score(name, p, S, load):
    prof = sum(r.kwh * S[r.shape] for r in p.itertuples()) / 1e3  # MW per hour
    kwh = p.kwh.sum()
    capex = p.capex_usd.sum()
    om = p.om_usd_yr.sum()
    lcoe = (capex * FIN["crf"] + om) / (kwh * FIN["levelized_energy_factor"])
    used = np.minimum(prof, load).sum() / 1e3
    pay = {k: capex / (kwh * v - om) if kwh * v > om else np.inf for k, v in FIN["electricity_price_usd_per_kwh"].items()}
    mix = p.groupby("land_type").kwh.sum() / kwh
    return dict(
        portfolio=name, sites=len(p), MWdc=p.kwdc.sum() / 1e3, GWh=kwh / 1e6, pct_campus=100 * kwh / 1e6 / CAMPUS,
        new_land_ha=p.new_land_m2.sum() / 1e4, built_area_ha=p.area_m2.sum() / 1e4, capex_musd=capex / 1e6,
        lcoe_c_kwh=100 * lcoe, lcoe_c_kwh_itc30=100 * (capex * (1 - FIN["itc_sensitivity"]) * FIN["crf"] + om) / (kwh * FIN["levelized_energy_factor"]),
        payback_yr_5_6c=pay["base_purchase_fy23"], payback_yr_10c=pay["ues_rate_fy27"],
        peak_MWac=prof.max(), used_on_campus_pct=100 * used / (kwh / 1e6), exported_GWh=kwh / 1e6 - used,
        mix=", ".join(f"{k} {100 * v:.0f}%" for k, v in mix.sort_values(ascending=False).items()),
        designs=", ".join(f"{k} ({v})" for k, v in p.design.value_counts().items()))


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower().replace("%", "pct")).strip("_")


def main():
    PD.mkdir(parents=True, exist_ok=True)
    o = site_options()
    o.to_csv(PD / "site_options.csv", index=False)
    S = shapes()
    load = cb.load_shapes(CAMPUS)["cooling"]
    P = build(o)
    P["D grid-limited (export <= 5 %)"] = grid_limited(o, S, load)
    rows = []
    for name, p in P.items():
        p.to_csv(PD / f"sites_{slug(name)}.csv", index=False)
        rows.append(score(name, p, S, load))
    s = pd.DataFrame(rows)
    s.round(2).to_csv(PD / "summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(f"[{V['name']}] campus FY2026 {CAMPUS} GWh; targets {TARGETS}; {o.site_id.nunique()} sites, {len(o)} options")
    cols = ["portfolio", "sites", "MWdc", "GWh", "pct_campus", "new_land_ha", "capex_musd", "lcoe_c_kwh",
            "lcoe_c_kwh_itc30", "payback_yr_5_6c", "payback_yr_10c", "peak_MWac", "used_on_campus_pct", "exported_GWh"]
    print(s[cols].round(1).to_string(index=False))
    print()
    print(s[["portfolio", "mix", "designs"]].to_string(index=False))


if __name__ == "__main__":
    main()
