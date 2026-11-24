"""Step 7: campus electricity balance - how much of the campus load on-site PV can carry (no storage).

  python 09_campus_balance.py --vintage 2025

Campus electricity (FY2026 estimate) = purchased (grid, incl. Samson PPA) + CHP. Evidence:
  FY2023 purchased 486.1 GWh (Tex. Gov. Code 2265 report, $27.2 M -> 5.6 c/kWh purchase price);
  CHP net generation from EIA-923 plant 58151 (PUDL), e.g. 2025 = 133.8 GWh;
  FY2022 'imported electricity' 445.7 GWh (AASHE STARS); FY2019 purchased ~317-360 GWh (GHG inventory).
  -> total ~525 (FY19), ~553 (FY22), ~611 GWh (FY23); FY2026 central 650 GWh (low 611, high 686).
CHP is held at its 2025 output and assumed flat (baseload); PV must meet the residual (load - CHP).

No hourly campus load is public, so two shapes for the residual load are compared:
  flat     : constant
  cooling  : base + k * cooling-degree-hours (TMY dry bulb > 18 C), peak/mean = 1.3
Outputs: outputs/<v>/campus_balance.csv
"""
import json

import numpy as np
import pandas as pd

from vintage import ROOT, get_vintage

V = get_vintage()
OUTD = V["out_dir"]
CFG = json.loads((ROOT / "config/sam_defaults.json").read_text())

CAMPUS_GWH = {"low": 611.0, "central": 650.0, "high": 686.0}
CHP_GWH = 133.8          # EIA-923, calendar 2025
PEAK_TO_MEAN = 1.3       # cooling-driven shape
T_BASE = 18.0            # C


def load_shapes(total_gwh):
    resid = (total_gwh - CHP_GWH) * 1e3  # MWh
    w = pd.read_csv(ROOT / CFG["weather_file"], skiprows=2)
    cdh = np.clip(w["Temperature"].values - T_BASE, 0, None)
    flat = np.full(8760, resid / 8760)
    # base + k*cdh with mean = resid/8760 and max = PEAK_TO_MEAN * mean
    mean = resid / 8760
    k = (PEAK_TO_MEAN - 1) * mean / (cdh.max() - cdh.mean())
    cool = mean + k * (cdh - cdh.mean())
    return {"flat": flat, "cooling": cool}


def main():
    H = pd.read_csv(OUTD / "energy_hourly_mw.csv", index_col="hour")
    built = ["roof", "garage canopy", "parking carport"]
    portfolios = {
        "roofs + garages": H[["roof", "garage canopy"]].sum(axis=1),
        "roofs + garages + parking": H[built].sum(axis=1),
        "+ all open land (fixed)": H[built].sum(axis=1) + H["open land (fixed)"],
        "+ all open land (tracker)": H[built].sum(axis=1) + H["open land (tracker)"],
    }
    rows = []
    for scen, total in CAMPUS_GWH.items():
        for shape, L in load_shapes(total).items():
            for name, pv in portfolios.items():
                p = pv.values
                direct = np.minimum(p, L).sum() / 1e3
                day = p > 0
                rows.append(dict(
                    campus_scenario=scen, campus_gwh=total, load_shape=shape, portfolio=name,
                    pv_gwh=round(p.sum() / 1e3, 1), pv_peak_mwac=round(p.max(), 1),
                    pv_pct_of_campus=round(100 * p.sum() / 1e3 / total, 1),
                    used_on_campus_gwh=round(direct, 1),
                    exported_gwh=round(np.maximum(p - L, 0).sum() / 1e3, 1),
                    grid_import_gwh=round(np.maximum(L - p, 0).sum() / 1e3, 1),
                    grid_cut_pct=round(100 * direct / ((total - CHP_GWH)), 1),
                    daytime_load_met_pct=round(100 * np.minimum(p, L)[day].sum() / L[day].sum(), 1),
                    hours_no_import=int((p >= L).sum())))
    df = pd.DataFrame(rows)
    df.to_csv(OUTD / "campus_balance.csv", index=False)
    print(f"campus FY2026 (GWh): {CAMPUS_GWH}; CHP held at {CHP_GWH} GWh -> residual (central) "
          f"{CAMPUS_GWH['central'] - CHP_GWH:.0f} GWh = {(CAMPUS_GWH['central'] - CHP_GWH) / 8.76:.1f} MW average")
    show = df[df.campus_scenario == "central"].drop(columns=["campus_scenario", "campus_gwh"])
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
