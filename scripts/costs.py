"""Site-level capital cost model (v2), shared by 11_portfolios.py and the web page (webapp/model.js
mirrors it - keep the two in sync).

  capex = kWdc * 1000 * base_usd_per_w[cost_key] * size_factor   + adders
  size_factor = clip((kWdc / ref_kwdc[cost_key]) ** -exponent, lo, hi)      economies of scale
  adders (all PLACEHOLDERS in config/sam_defaults.json -> costs.model):
    height   roofs / garages : kWdc*1000 * usd_per_w_per_m * max(height - free_m, 0)     crane, access
    reroof   roofs           : kWdc*1000 * usd_per_w  if building age >= age_yr          roof R&R in life
    garage   garage tops     : kWdc*1000 * usd_per_w                                      structural
    interconnection (parking, open land): kWdc < lv_max_kwdc -> lv_fixed_usd
                                          else mv_fixed_usd + mv_usd_per_m * distance to substation
    site prep (open land)    : usd_per_m2 * land m2                                       grading, fencing
"""
import numpy as np

COMPONENTS = ["base", "scale", "height", "reroof", "garage", "interconnection", "siteprep"]


def capex_breakdown(cost_key, land_type, kwdc, land_m2, height_m, age_yr, dist_m, costs):
    """Return dict of capex components [USD] for one site option."""
    M = costs["model"]
    w = kwdc * 1000.0
    base = w * costs["installed_usd_per_wdc"][cost_key]
    f = (max(kwdc, 1.0) / M["scale"]["ref_kwdc"][cost_key]) ** (-M["scale"]["exponent"])
    f = float(np.clip(f, M["scale"]["min_factor"], M["scale"]["max_factor"]))
    out = dict(base=base, scale=base * (f - 1), height=0.0, reroof=0.0, garage=0.0, interconnection=0.0, siteprep=0.0)
    if land_type in ("roof", "garage top") and height_m is not None and np.isfinite(height_m):
        out["height"] = w * M["height"]["usd_per_w_per_m"] * max(height_m - M["height"]["free_m"], 0.0)
    if land_type == "roof" and age_yr is not None and np.isfinite(age_yr) and age_yr >= M["reroof"]["age_yr"]:
        out["reroof"] = w * M["reroof"]["usd_per_w"]
    if land_type == "garage top":
        out["garage"] = w * M["garage"]["usd_per_w"]
    if land_type in ("parking", "open land"):
        I = M["interconnection"]
        d = dist_m if dist_m is not None and np.isfinite(dist_m) else I["default_dist_m"]
        out["interconnection"] = I["lv_fixed_usd"] if kwdc < I["lv_max_kwdc"] else I["mv_fixed_usd"] + I["mv_usd_per_m"] * d
    if land_type == "open land":
        out["siteprep"] = M["siteprep"]["usd_per_m2"] * land_m2
    return out


DEFAULT_MODEL = {
    "_status": "PLACEHOLDERS - plausible orders of magnitude, not from a cited source; replace with the team's cost table",
    "scale": {"exponent": 0.10, "min_factor": 0.85, "max_factor": 1.6,
              "ref_kwdc": {"roof": 500, "roof_east_west": 500, "carport": 500, "garage_canopy": 1000,
                           "ground_fixed": 5000, "ground_tracker": 5000},
              "_note": "base $/W applies at ref_kwdc; ~-20 % per 10x size (exponent 0.10)"},
    "height": {"usd_per_w_per_m": 0.005, "free_m": 12, "_note": "crane / material lift / access above ~3 storeys"},
    "reroof": {"usd_per_w": 0.20, "age_yr": 20,
               "_note": "remove & reinstall the array once during its life when the building (proxy for roof) is >= 20 yr old"},
    "garage": {"usd_per_w": 0.30, "_note": "structural review / reinforcement of the (post-tensioned) top deck"},
    "interconnection": {"lv_max_kwdc": 250, "lv_fixed_usd": 50000, "mv_fixed_usd": 150000, "mv_usd_per_m": 400,
                        "default_dist_m": 800,
                        "_note": "small sites tie into a nearby building service; larger ones need a pad-mount transformer + "
                                 "switchgear on the 12.47 kV loop and medium-voltage ductbank (distance to the nearest TAMU substation as proxy)"},
    "siteprep": {"usd_per_m2": 4.0, "_note": "grading, fencing, access on open land"},
}
