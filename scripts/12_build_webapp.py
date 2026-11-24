"""Step 10: data for the interactive web page (webapp/).

  python 12_build_webapp.py        then  ./serve.sh  ->  http://localhost:8000

Writes webapp/data/:
  basemap.jpg              30 cm imagery downsampled (~0.6 m), in the tile's UTM frame
  common.json              tile bounds, campus electricity, CHP, design profiles (8760, normalised),
                           residual campus load (load - CHP) for low / central / high, cost & finance defaults
  sites_<v>.json           per vintage: one entry per site with geometry (UTM metres relative to the tile
                           origin, simplified) and its design options (kWdc, kWh, area, new land, cost key)
The page re-implements the portfolio logic of 11_portfolios.py in JavaScript (webapp/model.js).
"""
import importlib.util
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.enums import Resampling

from vintage import ROOT

WEB = ROOT / "webapp/data"
X0, Y0 = 753000.0, 3388500.0
TILE = 3000.0
BASEMAP_PX = 5000
SIMPLIFY_M = 0.5
CFG = json.loads((ROOT / "config/sam_defaults.json").read_text())


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def basemap():
    with rasterio.open(ROOT / "tamu_land_resources/tif/merged_tamu_esri_30cm.tif") as src:
        img = src.read([1, 2, 3], out_shape=(3, BASEMAP_PX, BASEMAP_PX), resampling=Resampling.average)
    Image.fromarray(np.moveaxis(img, 0, -1)).save(WEB / "basemap.jpg", quality=82, optimize=True)


def coords(geom):
    """GeoJSON-like rings in metres relative to the tile origin, rounded to 0.1 m."""
    g = geom.simplify(SIMPLIFY_M, preserve_topology=True)
    polys = [g] if g.geom_type == "Polygon" else list(getattr(g, "geoms", []))
    out = []
    for p in polys:
        if p.is_empty or p.geom_type != "Polygon":
            continue
        rings = [p.exterior, *p.interiors]
        out.append([[[round(x - X0, 1), round(y - Y0, 1)] for x, y in r.coords] for r in rings])
    return out


def cost_key(land_type, design):
    return {"roof": "roof_east_west" if design.startswith("east-west") else "roof",
            "garage top": "garage_canopy", "parking": "carport"}.get(
        land_type, "ground_tracker" if design.startswith("tracker") else "ground_fixed")


def sites(vintage, pf):
    d = ROOT / f"outputs/{vintage}"
    pf.V = {**pf.V, "out_dir": d}
    pf.OUTD, pf.PD = d, d / "portfolios"
    pf.PD.mkdir(parents=True, exist_ok=True)
    o = pf.site_options()  # same options as 11_portfolios.py (incl. the open-land policy screen)
    inv = d / "inventory.gpkg"
    roofs = gpd.read_file(inv, layer="roofs").set_index("roof_id")
    canopies = gpd.read_file(inv, layer="parking_canopies")
    open_ = gpd.read_file(inv, layer="open_land").set_index("open_id")
    can_geom = canopies.dissolve("lot_id").geometry
    lot_geom = gpd.read_file(inv, layer="parking_lots").set_index("lot_id").geometry  # lots without stall lines
    out = []
    for sid, g in o.groupby("site_id", sort=False):
        lt = g.land_type.iloc[0]
        if lt in ("roof", "garage top"):
            r = roofs.loc[sid]
            geom = r.geometry
            info = dict(type=r.get("tamu_type"), year=None if pd.isna(r.get("year_built")) else int(r["year_built"]),
                        height_m=None if pd.isna(r.get("height_m")) else float(r["height_m"]),
                        footprint_m2=round(float(r["area_m2"])), shade_loss_pct=None if pd.isna(r.get("shade_loss_pct")) else float(r["shade_loss_pct"]),
                        source=r.get("usable_source"))
        elif lt == "parking":
            geom = can_geom.get(sid)
            info = {"canopy": "stall rows"}
            if geom is None:
                geom, info = lot_geom.get(sid), {"canopy": "estimate (no stall lines) - lot outline shown"}
        else:
            r = open_.loc[sid]
            geom = r.geometry
            info = dict(canopy_frac=float(r["canopy_frac"]), slope_deg=float(r["slope_mean_deg"]),
                        shade_loss_pct=None if pd.isna(r.get("shade_loss_pct")) else float(r["shade_loss_pct"]))
        if geom is None:
            continue
        num = lambda v: None if pd.isna(v) else round(float(v), 1)
        drivers = dict(height_m=num(g.height_m.iloc[0]), age_yr=num(g.age_yr.iloc[0]), dist_m=num(g.dist_m.iloc[0]))
        out.append(dict(
            id=sid, name=str(g.name.iloc[0]), land=lt, polo=bool(sid == pf.POLO), info=info, geom=coords(geom), cost=drivers,
            options=[dict(design=r.design, shape=r.shape, kwdc=round(r.kwdc, 1), kwh=round(r.kwh),
                          area_m2=round(r.area_m2, 1), new_land_m2=round(r.new_land_m2, 1),
                          cost_key=cost_key(lt, r.design)) for r in g.itertuples()]))
    (WEB / f"sites_{vintage}.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"{vintage}: {len(out)} sites, {sum(len(s['options']) for s in out)} options")


def common(pf, cb):
    S = pf.shapes()
    loads = {k: np.round(cb.load_shapes(v)["cooling"], 3).tolist() for k, v in cb.CAMPUS_GWH.items()}
    data = dict(
        tile=dict(x0=X0, y0=Y0, size=TILE, crs="EPSG:6343 (NAD83(2011) / UTM 14N)"),
        campus_gwh=cb.CAMPUS_GWH, chp_gwh=cb.CHP_GWH, residual_load_mw=loads,
        shapes={k: np.round(v * 1e6, 2).tolist() for k, v in S.items()},  # share of annual energy per hour x 1e6
        shape_scale=1e6,
        costs=dict(usd_per_wdc=CFG["costs"]["installed_usd_per_wdc"], om=CFG["costs"]["om_usd_per_kwdc_yr"],
                   finance=CFG["costs"]["finance"], model=CFG["costs"]["model"]),
        land_value_usd_per_acre=250000, max_export_frac=0.05)
    (WEB / "common.json").write_text(json.dumps(data, separators=(",", ":")))


def main():
    WEB.mkdir(parents=True, exist_ok=True)
    basemap()
    pf = load_module("pf", "11_portfolios.py")
    cb = load_module("cb", "09_campus_balance.py")
    common(pf, cb)
    for v in ("2025", "2017"):
        if (ROOT / f"outputs/{v}/energy_sections.csv").exists():
            sites(v, pf)
    for f in sorted(WEB.iterdir()):
        print(f"  {f.name:20s} {f.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
