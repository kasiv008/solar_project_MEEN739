"""Step 1 of the inventory: rooftop footprints + a separate open-land layer.

Rooftops: start from the TAMU "University Building" layer (the Aggie Map data), not OSM, and
apply rule-based screens that need no LiDAR. Every footprint is kept in `roofs_all` with the
first screen it failed, so nothing silently disappears.

Open land: what is left of the TAMU "Main Campus" area after removing buildings, pavement,
parking, water and fields. It is saved separately with land-use / tree attributes and only
light screening, for later use.

Output: data/interim/inventory_step1.gpkg  (layers: roofs_all, roofs_candidates, open_land_all,
open_land_candidates) and outputs/step1_roofs_vs_v2.csv (comparison with Team4_Portfolios_v2).
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import box

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/tamu_fcor.gpkg"
OSM = ROOT / "tamu_land_resources/osm/merged_tamu_osm.gpkg"
OUT = ROOT / "data/interim/inventory_step1.gpkg"
V2 = ROOT / "Team4_Portfolios_v2.xlsx"
CRS = 6343
TILE = box(753000, 3388500, 756000, 3391500)


# --- rooftop screening rules -------------------------------------------------------------
MIN_ROOF_M2 = 500
NOT_OWNED_TYPES = {"THE ASSOCIATION", "TAMU FOUNDATION"}  # in the university layer, owned by others
NO_ROOF_TYPES = {"BUS SHELTER", "CART SHELTER", "PICNIC SHELTER", "VEH SHELTER", "STORAGE", "MECH"}
# Open-air venues: footprint is a bowl / stands, no usable roof deck
OPEN_AIR = ["kyle field", "olsen field", "davis diamond", "softball stadium", "cushing stadium",
            "ellis field", "track & field complex", "amphitheater"]

# --- open-land screening rules -----------------------------------------------------------
MIN_OPEN_M2 = 5000
ROAD_HALF_WIDTH_M = 6      # county road centrelines -> carriageway
DRIVEWAY_HALF_WIDTH_M = 3  # campus driveways
BUILDING_BUFFER_M = 3      # keep arrays off building walls / fire access


def add_location(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """SAM/PySAM site inputs that come from geometry alone: lat/lon of an interior point.
    Elevation, tilt, azimuth, GCR, kWdc and shading are added in the LiDAR / design steps."""
    pt = g.geometry.representative_point()
    g["x_utm"], g["y_utm"] = pt.x.round(1), pt.y.round(1)
    ll = pt.to_crs(4326)
    g["lat"], g["lon"] = ll.y.round(6), ll.x.round(6)
    return g


def read(layer: str) -> gpd.GeoDataFrame:
    return gpd.read_file(RAW, layer=layer).to_crs(CRS)


def clean(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().replace("", pd.NA)


# =========================================================================================
# Rooftops
# =========================================================================================
def screen_roofs() -> gpd.GeoDataFrame:
    b = read("university_building")
    b = b.rename(columns={"Number": "tamu_num", "BldgName": "name", "Type": "tamu_type",
                          "NumFloors": "floors", "YearBuilt": "year_built", "status": "tamu_status",
                          "IsBldg": "is_bldg"})
    for c in ["tamu_num", "name", "tamu_type", "tamu_status", "is_bldg", "floors"]:
        b[c] = clean(b[c])
    b["year_built"] = pd.to_numeric(b["year_built"], errors="coerce").astype("Int64")
    b["floors"] = pd.to_numeric(b["floors"], errors="coerce").astype("Int64")
    # parts like 0468.1 .. 0468.4 belong to building 0468
    b["bldg_id"] = b["tamu_num"].str.split(".").str[0]
    b["area_m2"] = b.area.round(1)
    b["perimeter_m"] = b.length.round(1)
    b["compactness"] = (4 * np.pi * b.area / b.length**2).round(3)  # 1 = circle, low = slivers
    # LiDAR and imagery only cover the tile
    b["frac_in_tile"] = (b.intersection(TILE).area / b.area).round(3)

    name = b["name"].fillna("").str.lower()
    reason = pd.Series(pd.NA, index=b.index, dtype="string")

    def drop(mask, why):
        reason.loc[mask & reason.isna()] = why

    drop(b["tamu_type"].isin(NOT_OWNED_TYPES), "affiliated entity, not university-owned")
    drop(b["tamu_type"].isin(NO_ROOF_TYPES), "shelter / mechanical / storage (no usable roof)")
    drop(b["is_bldg"].eq("N"), "not a building (TAMU IsBldg = N)")
    drop(name.str.contains("|".join(OPEN_AIR), regex=True), "open-air venue (no roof deck)")
    drop(b["tamu_type"].eq("CONSTRUCTION"), "under construction in TAMU layer - check imagery")
    drop(b["frac_in_tile"] < 0.5, "mostly outside the LiDAR / imagery tile")
    drop(b["area_m2"] < MIN_ROOF_M2, f"too small (< {MIN_ROOF_M2} m2)")
    b["screen_reason"] = reason.fillna("candidate")
    b["is_candidate"] = reason.isna()

    flags = []
    flags.append(np.where(b["name"].isna(), "unnamed footprint", ""))
    flags.append(np.where(b["frac_in_tile"].between(0.5, 0.999), "partly outside tile", ""))
    b["qa_flags"] = ["; ".join(f for f in fs if f) for fs in zip(*flags)]

    keep = ["bldg_id", "tamu_num", "name", "tamu_type", "year_built", "floors", "tamu_status",
            "is_bldg", "area_m2", "perimeter_m", "compactness", "frac_in_tile", "is_candidate", "screen_reason",
            "qa_flags", "geometry"]
    b = add_location(b)
    keep[-1:-1] = ["x_utm", "y_utm", "lat", "lon"]
    b = b[keep].sort_values("area_m2", ascending=False).reset_index(drop=True)
    b.insert(0, "roof_id", [f"R{i:04d}" for i in range(1, len(b) + 1)])
    return b


def compare_with_v2(roofs: gpd.GeoDataFrame) -> pd.DataFrame:
    """Join our screen with the team's v2 inventory by TAMU building number."""
    v2 = pd.read_excel(V2, "Roofs (ownership)")
    v2 = v2[v2["tamu_num"].notna()].copy()
    v2["bldg_id"] = v2["tamu_num"].astype(str).str.split(".").str[0].str.zfill(4)
    v2 = (v2.groupby("bldg_id")
            .agg(v2_area_m2=("area_m2", "sum"), v2_class=("roof_class", "first"),
                 v2_usable_m2=("usable_m2", "sum"))
            .reset_index())
    v2["v2_candidate"] = v2["v2_class"].str.contains("candidate|built after", regex=True)

    ours = (roofs[roofs["bldg_id"].notna()]
            .groupby("bldg_id")
            .agg(name=("name", "first"), tamu_type=("tamu_type", "first"),
                 year_built=("year_built", "max"), area_m2=("area_m2", "sum"),
                 is_candidate=("is_candidate", "any"), screen_reason=("screen_reason", "first"))
            .reset_index())
    m = ours.merge(v2, on="bldg_id", how="outer", indicator=True)
    m["is_candidate"] = m["is_candidate"].astype("boolean").fillna(False)
    m["v2_candidate"] = m["v2_candidate"].astype("boolean").fillna(False)
    m["change"] = np.select(
        [m["is_candidate"] & m["v2_candidate"],
         m["is_candidate"] & m["_merge"].eq("left_only"),
         m["is_candidate"] & ~m["v2_candidate"],
         ~m["is_candidate"] & m["v2_candidate"]],
        ["in both", "new: no OSM outline in v2", "new: v2 dropped it (LiDAR/size/owner)",
         "v2 only: dropped here"],
        default="out in both")
    return m.drop(columns="_merge").sort_values(["change", "area_m2"], ascending=[True, False])


# =========================================================================================
# Open land (kept separately)
# =========================================================================================
def build_open_land(roofs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    campus = read("campus_background")
    campus = campus[campus["Name"].eq("Main Campus")].union_all().intersection(TILE)

    hard = []  # everything that is not open ground
    for lyr in ["university_building", "non_university_building", "off_campus_building",
                "structure_data"]:
        hard.append(read(lyr).buffer(BUILDING_BUFFER_M))
    for lyr in ["surface_parking", "private_parking_lot", "garage_parking", "street_pavement",
                "sidewalk", "pond", "fountain", "electrical_substation", "deck", "gravel",
                "wall_area", "ramp_or_curb_cut", "fire_lane", "railroad_crossing"]:
        hard.append(read(lyr).geometry)
    hard.append(read("brazos_county_road_centerline").buffer(ROAD_HALF_WIDTH_M))
    hard.append(read("campus_driveway").buffer(DRIVEWAY_HALF_WIDTH_M))
    hard.append(read("railroad").buffer(5))
    hard_union = gpd.GeoSeries(pd.concat(hard, ignore_index=True), crs=CRS).make_valid().union_all()

    residual = gpd.GeoSeries([campus.difference(hard_union)], crs=CRS).explode(index_parts=False)
    # open away thin slivers (medians, strips between sidewalks): erode then dilate 5 m
    residual = residual.buffer(-5).buffer(5).explode(index_parts=False)
    o = gpd.GeoDataFrame(geometry=residual[residual.area >= 1000].values, crs=CRS)
    o["area_m2"] = o.area.round(1)

    # --- attributes: protected uses, trees, OSM tags --------------------------------------
    def overlap_frac(other: gpd.GeoDataFrame, label_col: str | None, tag: str):
        other = other[other.geometry.notna()].copy()
        inter = gpd.overlay(o.reset_index()[["index", "geometry"]],
                            other[[c for c in [label_col, "geometry"] if c]], how="intersection",
                            keep_geom_type=True)
        if inter.empty:
            o[f"{tag}_frac"] = 0.0
            if label_col:
                o[f"{tag}_name"] = pd.NA
            return
        inter["a"] = inter.area
        o[f"{tag}_frac"] = (inter.groupby("index")["a"].sum() / o["area_m2"]).reindex(o.index).fillna(0).round(3)
        if label_col:
            top = inter.sort_values("a").groupby("index")[label_col].last()
            o[f"{tag}_name"] = top.reindex(o.index)

    lu = read("land_use")
    lu["lu_label"] = lu["LandID"].astype(str) + ": " + lu["Name"].astype(str)
    overlap_frac(lu, "lu_label", "landuse")
    overlap_frac(read("park_and_garden"), "Name", "park")
    overlap_frac(read("golf_course"), None, "golf")
    fields = pd.concat([read(l) for l in ["sports_and_rec_field_area", "field", "field_30",
                                          "field_33", "field_36", "field_39", "field_42",
                                          "tennis_court", "end_zone", "mid_field"]],
                       ignore_index=True)
    overlap_frac(gpd.GeoDataFrame(fields, crs=CRS), None, "sports")

    trees = read("trees")
    t = gpd.sjoin(trees[["geometry"]], o[["geometry"]], predicate="within")
    o["tree_count"] = t.groupby("index_right").size().reindex(o.index).fillna(0).astype(int)
    o["trees_per_ha"] = (o["tree_count"] / (o["area_m2"] / 1e4)).round(1)

    osm = gpd.read_file(OSM, layer="multipolygons").to_crs(CRS)
    osm["osm_tag"] = (osm["landuse"].fillna("") + "|" + osm["leisure"].fillna("")).str.strip("|")
    osm = osm[osm["osm_tag"] != ""]
    overlap_frac(osm[["osm_tag", "geometry"]].assign(geometry=lambda d: d.make_valid()), "osm_tag", "osm")

    # --- light screening -----------------------------------------------------------------
    reason = pd.Series(pd.NA, index=o.index, dtype="string")

    def drop(mask, why):
        reason.loc[mask & reason.isna()] = why

    drop(o["golf_frac"] >= 0.2, "golf course")
    drop(o["sports_frac"] >= 0.2, "sports / rec field")
    drop(o["park_frac"] >= 0.2, "campus park / quad / memorial")
    drop(o["landuse_frac"] >= 0.5, "TAMU land-use record (rec sports, park, memorial)")
    drop(o["area_m2"] < MIN_OPEN_M2, f"too small (< {MIN_OPEN_M2} m2)")
    o["screen_reason"] = reason.fillna("candidate")
    o["is_candidate"] = reason.isna()
    o["note"] = np.where(o["trees_per_ha"] > 40, "many trees - check canopy with LiDAR", "")

    o = add_location(o)
    o = o.sort_values("area_m2", ascending=False).reset_index(drop=True)
    o.insert(0, "open_id", [f"O{i:04d}" for i in range(1, len(o) + 1)])
    return o


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.unlink(missing_ok=True)
    (ROOT / "outputs").mkdir(exist_ok=True)

    roofs = screen_roofs()
    roofs.to_file(OUT, layer="roofs_all")
    roofs[roofs["is_candidate"]].to_file(OUT, layer="roofs_candidates")
    print("ROOFS  (TAMU University Building layer, tile)")
    print(roofs.groupby("screen_reason")["area_m2"].agg(n="count", m2="sum").round(0)
          .sort_values("m2", ascending=False).to_string(), "\n")
    c = roofs[roofs["is_candidate"]]
    print(f"candidates: {len(c)} parts / {c['bldg_id'].nunique()} buildings, "
          f"{c['area_m2'].sum():,.0f} m2 footprint")

    cmp_ = compare_with_v2(roofs)
    cmp_.to_csv(ROOT / "outputs/step1_roofs_vs_v2.csv", index=False)
    print("\nvs Team4_Portfolios_v2 (by building number):")
    print(cmp_.groupby("change")["area_m2"].agg(n="count", m2="sum").round(0).to_string())

    o = build_open_land(roofs)
    o.to_file(OUT, layer="open_land_all")
    o[o["is_candidate"]].to_file(OUT, layer="open_land_candidates")
    print("\nOPEN LAND (Main Campus minus buildings, pavement, parking, water)")
    print(o.groupby("screen_reason")["area_m2"].agg(n="count", m2="sum").round(0).to_string())


if __name__ == "__main__":
    main()
