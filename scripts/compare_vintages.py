"""Per-roof comparison of the 2017 and 2025 LiDAR inventories -> outputs/compare_2017_vs_2025.csv"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
cols = ["roof_id", "final_usable_m2", "system_capacity_kw", "shade_loss_pct", "flat_frac", "rough_frac",
        "height_m", "usable_source"]
a = pd.read_csv(ROOT / "outputs/2017/roofs.csv")
b = pd.read_csv(ROOT / "outputs/2025/roofs.csv")
m = a[["roof_id", "bldg_id", "name", "tamu_type", "year_built", "area_m2"] + cols[1:]].merge(
    b[cols], on="roof_id", suffixes=("_2017", "_2025"))
m["d_kw"] = (m.system_capacity_kw_2025 - m.system_capacity_kw_2017).round(1)
m["d_pct"] = (100 * m.d_kw / m.system_capacity_kw_2017.replace(0, np.nan)).round(0)
m["why"] = np.select(
    [m.usable_source_2017.str.startswith("estimate") & m.usable_source_2025.str.startswith("LiDAR"),
     m.usable_source_2025.str.startswith("estimate"),
     m.d_pct.abs() >= 25],
    ["2017 estimate -> measured in 2025", "still estimated in 2025", "large change - check imagery"],
    default="")
m.sort_values("d_kw", key=abs, ascending=False).to_csv(ROOT / "outputs/compare_2017_vs_2025.csv", index=False)
both = m.usable_source_2017.str.startswith("LiDAR") & m.usable_source_2025.str.startswith("LiDAR")
print(f"roofs measured in both: {both.sum()}; capacity {m.loc[both, 'system_capacity_kw_2017'].sum() / 1e3:.1f} -> "
      f"{m.loc[both, 'system_capacity_kw_2025'].sum() / 1e3:.1f} MWdc; correlation "
      f"{m.loc[both, ['system_capacity_kw_2017', 'system_capacity_kw_2025']].corr().iloc[0, 1]:.3f}; "
      f"median change {m.loc[both, 'd_pct'].median():+.0f} %")
print(m.why.value_counts().to_string())
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 36)
print(m.sort_values("d_kw", key=abs, ascending=False)[
    ["roof_id", "name", "year_built", "system_capacity_kw_2017", "system_capacity_kw_2025", "d_pct", "why"]].head(15).to_string(index=False))
