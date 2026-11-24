"""Download TAMU FCOR basemap layers (the data behind Aggie Map) for the campus tile.

Output: data/raw/tamu_fcor.gpkg, one GeoPackage layer per service layer, in EPSG:6343
(NAD83(2011) / UTM 14N, the same grid as the LiDAR and the 30 cm imagery).
"""
import json
import re
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import (LinearRing, MultiLineString, MultiPoint, MultiPolygon,
                              Point, Polygon)
from shapely.validation import make_valid

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/tamu_fcor.gpkg"
STATUS = ROOT / "data/raw/tamu_fcor_status.csv"  # per-layer fetched vs expected counts

SERVICE = "https://gis.tamu.edu/arcgis/rest/services/FCOR/TAMU_Basemap_20261008/MapServer"
CRS = 6343
TILE = dict(xmin=753000, ymin=3388500, xmax=756000, ymax=3391500)  # same 3 km tile as LiDAR / TIF


def list_layers() -> dict[int, str]:
    """All feature layers of the basemap (group layers have no geometry and are skipped)."""
    js = requests.get(SERVICE, params=dict(f="json"), timeout=60).json()
    out = {}
    for lyr in js["layers"]:
        if not lyr.get("geometryType"):
            continue
        name = re.sub(r"[^a-z0-9]+", "_", lyr["name"].lower()).strip("_")
        if name in out.values():  # e.g. several "Field" layers under different stadiums
            name = f"{name}_{lyr['id']}"
        out[lyr["id"]] = name
    return out


def _get(url: str, params: dict, tries: int = 4) -> dict:
    """POST a query; the TAMU server fails intermittently, so retry with backoff."""
    for k in range(tries):
        try:
            r = requests.post(url, data=params, timeout=180)
            r.raise_for_status()
            js = r.json()
            if "error" not in js:
                return js
            err = js["error"]
        except (requests.RequestException, ValueError) as e:
            err = e
        time.sleep(2 * (k + 1))
    raise RuntimeError(err)


def esri_to_shapely(g: dict | None):
    """Convert an Esri JSON geometry (points, paths, rings) to shapely, keeping its CRS."""
    if not g:
        return None
    if "x" in g:
        return None if g["x"] is None else Point(g["x"], g["y"])
    if "points" in g:
        return MultiPoint(g["points"])
    if "paths" in g:
        return MultiLineString([p for p in g["paths"] if len(p) > 1])
    if "rings" in g:
        # Esri: outer rings clockwise, holes counter-clockwise
        shells, holes = [], []
        for r in g["rings"]:
            if len(r) < 4:
                continue
            (holes if LinearRing(r).is_ccw else shells).append(r)
        polys = [Polygon(sh) for sh in shells]
        for h in holes:
            hp = Polygon(h)
            for i, p in enumerate(polys):
                if p.contains(hp.representative_point()):
                    polys[i] = Polygon(p.exterior.coords, [*map(lambda x: x.coords, p.interiors), h])
                    break
        geom = MultiPolygon(polys) if len(polys) > 1 else (polys[0] if polys else None)
        return make_valid(geom) if geom is not None and not geom.is_valid else geom
    return None


def _fetch_ids(url: str, ids: list[int]) -> list[dict]:
    """Fetch features by object ID; halve the batch when the server chokes on it."""
    try:
        js = _get(url, dict(objectIds=",".join(map(str, ids)), outFields="*",
                            returnGeometry="true", outSR=CRS, f="json"))
        return js.get("features", [])
    except RuntimeError:
        if len(ids) == 1:
            print(f"    skipped object {ids[0]} (server error)")
            return []
        m = len(ids) // 2
        return _fetch_ids(url, ids[:m]) + _fetch_ids(url, ids[m:])


def query(layer_id: int, chunk: int = 100) -> tuple[gpd.GeoDataFrame, int]:
    """Fetch all features of one layer intersecting the tile.

    The server caps results at 2000 and does not support pagination, so we ask for the
    object IDs first and then download them in batches as Esri JSON in EPSG:6343.
    """
    url = f"{SERVICE}/{layer_id}/query"
    env = dict(geometry=json.dumps({**TILE, "spatialReference": {"wkid": CRS}}),
               geometryType="esriGeometryEnvelope", inSR=CRS, spatialRel="esriSpatialRelIntersects")
    try:
        ids = _get(url, dict(where="1=1", returnIdsOnly="true", f="json", **env))["objectIds"]
    except RuntimeError:  # some layers reject spatial filters; take everything and clip below
        ids = _get(url, dict(where="1=1", returnIdsOnly="true", f="json"))["objectIds"]
    ids = sorted(ids or [])
    n_expected = len(ids)
    feats = []
    for i in range(0, len(ids), chunk):
        feats += _fetch_ids(url, ids[i:i + chunk])
    if not feats:
        return gpd.GeoDataFrame(geometry=[], crs=CRS), n_expected
    gdf = gpd.GeoDataFrame(pd.DataFrame([f["attributes"] for f in feats]),
                           geometry=[esri_to_shapely(f.get("geometry")) for f in feats], crs=CRS)
    for c in gdf.columns:  # Esri dates come as epoch milliseconds
        if c != "geometry" and ("date" in c.lower()) and pd.api.types.is_numeric_dtype(gdf[c]):
            gdf[c] = pd.to_datetime(gdf[c], unit="ms", errors="coerce")
    gdf.attrs["n_fetched"] = len(feats)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    return gdf.cx[TILE["xmin"]:TILE["xmax"], TILE["ymin"]:TILE["ymax"]], n_expected


def main():
    """Download every layer; on re-runs only layers that are missing or incomplete are fetched."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    status = pd.read_csv(STATUS, index_col="layer") if STATUS.exists() else pd.DataFrame()
    for lid, name in list_layers().items():
        if name in status.index and status.loc[name, "complete"]:
            continue
        try:
            gdf, n_expected = query(lid)
        except Exception as e:  # keep going; report which layers failed
            print(f"{lid:>3} {name:<34} FAILED: {e}")
            continue
        gdf.to_file(OUT, layer=name, driver="GPKG")  # overwrites the layer, keeps the others
        n_got = int(gdf.attrs.get("n_fetched", len(gdf)))
        status.loc[name, ["id", "fetched", "expected", "complete"]] = [lid, n_got, n_expected, n_got >= n_expected]
        status.index.name = "layer"
        status.to_csv(STATUS)
        flag = "" if n_got >= n_expected else f"  INCOMPLETE ({n_expected - n_got} missing)"
        print(f"{lid:>3} {name:<34} {len(gdf):>6} features in tile{flag}")
    bad = status[~status["complete"].astype(bool)]
    print(f"\n{len(status) - len(bad)} layers complete, {len(bad)} incomplete: {list(bad.index)}")


if __name__ == "__main__":
    main()
