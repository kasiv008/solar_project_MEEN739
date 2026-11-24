"""Grid one LiDAR vintage into terrain and surface models:  python 03_lidar_rasters.py --vintage 2017|2025

2017 (USGS 3DEP): only class 1 / 2, so buildings and trees are separated by height (nDSM).
2025 (StratMap): classified (6 = building), so a building-only surface is written as well.
Tiles may extend past the campus tile; only points inside it are used.

Outputs (data/interim/lidar_<v>/, 0.5 m grid aligned with the tile, EPSG:6343, NAVD88 metres):
  dtm.tif        bare-earth elevation (ground points, holes under buildings interpolated)
  dsm_all.tif    highest point of any return      -> roofs + trees + equipment (used for shading)
  dsm_last.tif   highest last return              -> roof surface with tree crowns mostly removed
  dsm_bldg.tif   highest building-class point     -> roof surface (only if the data has class 6)
  npts.tif       points per cell (data coverage check)
  multi.tif      share of points in the cell that are part of a multi-return pulse (vegetation cue)
"""
import laspy
import numpy as np
import rasterio
from rasterio.fill import fillnodata
from rasterio.transform import from_origin
from scipy import ndimage

from vintage import get_vintage

V = get_vintage()
LAZ_FILES = V["laz_files"]
OUT = V["lidar_dir"]

X0, Y0, X1, Y1 = 753000.0, 3388500.0, 756000.0, 3391500.0
RES = 0.5
NX, NY = int((X1 - X0) / RES), int((Y1 - Y0) / RES)
TRANSFORM = from_origin(X0, Y1, RES, RES)
NOISE_CLASSES = (7, 14, 18)  # low / high noise, wire conductors
BUILDING_CLASS = 6
MAX_ABOVE_GROUND = 120.0  # tallest campus structures are < 80 m; above this = birds / noise
MAX_BELOW_GROUND = 2.0


def cell_index(x, y):
    col = ((x - X0) / RES).astype(np.int64)
    row = ((Y1 - y) / RES).astype(np.int64)
    ok = (col >= 0) & (col < NX) & (row >= 0) & (row < NY)
    return row * NX + col, ok


def chunks():
    """Points inside the campus tile, from every LAZ file of the vintage."""
    for path in LAZ_FILES:
        with laspy.open(path) as f:
            (mx, my, _), (Mx, My, _) = f.header.mins, f.header.maxs
            if Mx < X0 or mx > X1 or My < Y0 or my > Y1:
                continue
            for pts in f.chunk_iterator(4_000_000):
                x, y = np.asarray(pts.x), np.asarray(pts.y)
                m = (x >= X0) & (x < X1) & (y >= Y0) & (y < Y1)
                if m.any():
                    yield (x[m], y[m], np.asarray(pts.z, dtype=np.float32)[m],
                           np.asarray(pts.classification)[m], np.asarray(pts.return_number)[m],
                           np.asarray(pts.number_of_returns)[m])


def write(name, arr, nodata=np.nan):
    with rasterio.open(OUT / f"{name}.tif", "w", driver="GTiff", height=NY, width=NX, count=1,
                       dtype=arr.dtype, crs="EPSG:6343", transform=TRANSFORM, nodata=nodata,
                       compress="deflate", predictor=3 if arr.dtype.kind == "f" else 2,
                       tiled=True) as dst:
        dst.write(arr, 1)


def build_dtm():
    """Mean of ground points per 1 m cell, holes filled by inverse-distance interpolation,
    then bilinear to 0.5 m (ground under large buildings is a smooth interpolation)."""
    nx2, ny2 = NX // 2, NY // 2
    s = np.zeros(nx2 * ny2, np.float64)
    n = np.zeros(nx2 * ny2, np.int32)
    for x, y, z, c, *_ in chunks():
        g = c == 2
        col = ((x[g] - X0) / 1.0).astype(np.int64)
        row = ((Y1 - y[g]) / 1.0).astype(np.int64)
        ok = (col >= 0) & (col < nx2) & (row >= 0) & (row < ny2)
        idx = row[ok] * nx2 + col[ok]
        s += np.bincount(idx, z[g][ok], minlength=s.size)
        n += np.bincount(idx, minlength=n.size).astype(np.int32)
    dtm1 = np.where(n > 0, s / np.maximum(n, 1), np.nan).reshape(ny2, nx2).astype(np.float32)
    # drop isolated ground outliers (mis-classified points) before filling
    med = ndimage.median_filter(np.nan_to_num(dtm1, nan=np.nanmedian(dtm1)), size=5)
    dtm1[np.abs(dtm1 - med) > 3] = np.nan
    mask = ~np.isnan(dtm1)
    filled = fillnodata(np.nan_to_num(dtm1), mask=mask.astype(np.uint8), max_search_distance=300,
                        smoothing_iterations=2)
    print(f"DTM: {100 * (1 - mask.mean()):.1f} % of 1 m cells had no ground points (filled)")
    return ndimage.zoom(filled, 2, order=1)[:NY, :NX].astype(np.float32)


def build_dsm(dtm):
    flat_dtm = dtm.ravel()
    dsm_all = np.full(NX * NY, -np.inf, np.float32)
    dsm_last = np.full(NX * NY, -np.inf, np.float32)
    dsm_bldg = np.full(NX * NY, -np.inf, np.float32)
    npts = np.zeros(NX * NY, np.int32)
    nmulti = np.zeros(NX * NY, np.int32)
    dropped = 0
    for x, y, z, c, rn, nr in chunks():
        idx, ok = cell_index(x, y)
        hag = z - flat_dtm[np.where(ok, idx, 0)]
        keep = ok & ~np.isin(c, NOISE_CLASSES) & (hag > -MAX_BELOW_GROUND) & (hag < MAX_ABOVE_GROUND)
        dropped += (~keep & ok).sum()
        i, zz = idx[keep], z[keep]
        np.maximum.at(dsm_all, i, zz)
        last = (rn[keep] == nr[keep])
        np.maximum.at(dsm_last, i[last], zz[last])
        b = c[keep] == BUILDING_CLASS
        np.maximum.at(dsm_bldg, i[b], zz[b])
        npts += np.bincount(i, minlength=npts.size).astype(np.int32)
        nmulti += np.bincount(i[nr[keep] > 1], minlength=npts.size).astype(np.int32)
    print(f"DSM: dropped {dropped:,} noise / out-of-range points")

    out = {}
    surfaces = [("dsm_all", dsm_all), ("dsm_last", dsm_last)]
    if np.isfinite(dsm_bldg).any():
        surfaces.append(("dsm_bldg", dsm_bldg))
    for name, d in surfaces:
        d = d.reshape(NY, NX)
        d[np.isinf(d)] = np.nan
        empty = np.isnan(d)
        # ~4 pts/m2 -> about a third of 0.5 m cells are empty; fill from neighbours (<= 1.5 m)
        d = fillnodata(np.nan_to_num(d), mask=(~empty).astype(np.uint8), max_search_distance=3)
        d[d == 0] = np.nan
        d = np.where(np.isnan(d), dtm, d)  # remaining gaps (water, dark roofs) -> ground
        # despike: a cell > 3 m above its 3x3 median with at most one neighbour near its height
        # (birds, wires). Roof corners and tree crowns have several such neighbours and stay.
        med = ndimage.median_filter(d, size=3)
        pad = np.pad(d, 1, mode="edge")
        close = sum((pad[1 + dr:1 + dr + NY, 1 + dc:1 + dc + NX] > d - 1.0).astype(np.int8)
                    for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
        spike = (d - med > 3) & (close <= 1)
        d[spike] = med[spike]
        d = np.maximum(d, dtm)  # surface never below ground
        print(f"{name}: {100 * empty.mean():.1f} % empty cells filled, {spike.sum():,} spikes removed")
        out[name] = d.astype(np.float32)
    npts = npts.reshape(NY, NX)
    multi = np.where(npts > 0, nmulti.reshape(NY, NX) / np.maximum(npts, 1), 0).astype(np.float32)
    return out, npts, multi


def main():
    dtm = build_dtm()
    write("dtm", dtm)
    print(f"vintage {V['name']}: {len(LAZ_FILES)} LAZ file(s)")
    surfaces, npts, multi = build_dsm(dtm)
    for name, d in surfaces.items():
        write(name, d)
    dsm_all = surfaces["dsm_all"]
    write("npts", npts.astype(np.int16), nodata=-1)
    write("multi", multi)
    ndsm = dsm_all - dtm
    print(f"nDSM > 2 m: {100 * (ndsm > 2).mean():.1f} % of the tile; "
          f"max {np.nanmax(ndsm):.1f} m; mean density {npts.mean() / RES**2:.2f} pts/m2")


if __name__ == "__main__":
    main()
