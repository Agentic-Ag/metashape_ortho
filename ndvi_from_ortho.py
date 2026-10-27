"""NDVI + band-split post-step for M3M multispectral orthomosaics.

Reads a multi-band orthomosaic exported by metashape_ortho and produces:
  --ndvi-out   float32 NDVI GeoTIFF, (NIR - Red) / (NIR + Red)
  --rgb-out    3-band RGB GeoTIFF (bands 1-3 of a combined M3M ortho)
  --ms-out     multi-band GeoTIFF with only the MS bands (G/R/RE/NIR)

Band roles are auto-detected from band descriptions where possible
(Metashape names M3M slave layers Green/Red/Red edge/NIR), else assumed
by position: a combined rig ortho is [R,G,B | G,R,RE,NIR] = 7 bands
(NIR=7, Red=5); an MS-only ortho is [G,R,RE,NIR] = 4 bands (NIR=4, Red=2).
Override with --nir-band / --red-band / --rgb-bands.

Usage:
    python ndvi_from_ortho.py ortho.tif --ndvi-out ndvi.tif
    python ndvi_from_ortho.py ortho.tif --rgb-out rgb.tif --ms-out ms.tif
    python ndvi_from_ortho.py ortho.tif            # print band table only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio


def _detect_bands(ds) -> tuple[int | None, int | None]:
    """(nir_band, red_band) 1-based, detected from band descriptions."""
    nir = red = None
    for i, desc in enumerate(ds.descriptions, start=1):
        d = (desc or "").lower()
        if "nir" in d or "near" in d:
            nir = i
        elif "red edge" in d or "rededge" in d or d == "re":
            pass                      # not NDVI red
        elif "red" in d:
            red = i
    return nir, red


def _default_bands(count: int) -> tuple[int, int]:
    """Positional fallback: NIR is last band, Red is 2 before it
    (M3M order G,R,RE,NIR with or without a leading RGB master trio)."""
    if count >= 4:
        return count, count - 2
    raise ValueError(f"Need >=4 bands for NDVI, ortho has {count}")


def _copy_profile(ds, count: int, dtype: str) -> dict:
    p = dict(ds.profile)
    p.update(count=count, dtype=dtype)
    p.pop("nodata", None)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ortho", help="Multispectral orthomosaic GeoTIFF")
    ap.add_argument("--ndvi-out", default=None)
    ap.add_argument("--rgb-out", default=None)
    ap.add_argument("--ms-out", default=None)
    ap.add_argument("--nir-band", type=int, default=None)
    ap.add_argument("--red-band", type=int, default=None)
    ap.add_argument("--rgb-bands", default="1,2,3")
    args = ap.parse_args()

    ortho = Path(args.ortho)
    if not ortho.exists():
        sys.exit(f"Not found: {ortho}")

    with rasterio.open(ortho) as ds:
        print(f"{ortho.name}: {ds.count} band(s), {ds.dtypes[0]}, "
              f"{ds.width}x{ds.height}, CRS {ds.crs}")
        for i, desc in enumerate(ds.descriptions, start=1):
            print(f"  band {i}: {desc or '(no name)'} {ds.dtypes[i-1]}")

        nir, red = args.nir_band, args.red_band
        if nir is None or red is None:
            det_nir, det_red = _detect_bands(ds)
            nir = nir or det_nir
            red = red or det_red
        if nir is None or red is None:
            nir, red = _default_bands(ds.count)

        need = args.ndvi_out or args.ms_out
        if need and max(nir, red) > ds.count:
            sys.exit(f"Band index out of range: nir={nir} red={red} "
                     f"but ortho has {ds.count} band(s)")

        if args.ndvi_out:
            n = ds.read(nir).astype(np.float32)
            r = ds.read(red).astype(np.float32)
            ndvi = np.where((n + r) != 0, (n - r) / (n + r), np.nan).astype(np.float32)
            prof = _copy_profile(ds, 1, "float32")
            prof.update(nodata=float("nan"))
            with rasterio.open(args.ndvi_out, "w", **prof) as out:
                out.write(ndvi, 1)
                out.set_band_description(1, "NDVI")
            print(f"NDVI (bands {nir},{red}) → {args.ndvi_out} "
                  f"[min {np.nanmin(ndvi):.3f} max {np.nanmax(ndvi):.3f}]")

        if args.rgb_out:
            bands = [int(b) for b in args.rgb_bands.split(",")]
            arr = ds.read(bands)
            prof = _copy_profile(ds, len(bands), ds.dtypes[0])
            with rasterio.open(args.rgb_out, "w", **prof) as out:
                out.write(arr)
            print(f"RGB (bands {bands}) → {args.rgb_out}")

        if args.ms_out:
            bands = [i for i in range(1, ds.count + 1)
                     if i not in {int(b) for b in args.rgb_bands.split(",")}] \
                or list(range(1, ds.count + 1))
            arr = ds.read(bands)
            prof = _copy_profile(ds, len(bands), ds.dtypes[0])
            with rasterio.open(args.ms_out, "w", **prof) as out:
                out.write(arr)
            print(f"MS (bands {bands}) → {args.ms_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
