"""Bridge between the Drone Pipeline and the Metashape ortho processor.

The Drone Pipeline (Backend/Mapping/Drone Pipeline) calls whatever is set in
config.yaml `orthomosaic.command` with {input_dir}/{output_dir} substituted.
This adapter:

  1. Stages the dropped imagery into the layout metashape_ortho expects:
         <staging>/<flight>_unprocessed/photos/<flight>/  (hardlinked)
  2. Writes a temp pipeline config (JSON — valid YAML, and parseable by
     Metashape's bundled Python which lacks PyYAML).
  3. Runs Metashape headless:  metashape.exe -r main.py <cfg> --yes
  4. Copies results into <output>:
         orthophoto.tif       — 3-band RGB ortho (pipeline classifies 'rgb')
         orthophoto_ms.tif    — multi-band ortho (pipeline classifies 'ms')
         ndvi.tif             — NDVI float raster (from ndvi_from_ortho.py)

Env overrides:
    METASHAPE_EXE   path to metashape.exe
                    (default C:\\Program Files\\Agisoft\\Metashape\\metashape.exe)
    METASHAPE_CRS   output CRS token (default EPSG::28356)
    AEROPOINTS_CSV  AeroPoints/Propeller GCP CSV to import (optional)

Usage (also what config.yaml's orthomosaic.command should be):
    python pipeline_adapter.py --input "{input_dir}" --output "{output_dir}"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger("metashape_adapter")

HERE = Path(__file__).resolve().parent
MAIN_PY = HERE / "main.py"
NDVI_PY = HERE / "ndvi_from_ortho.py"
METASHAPE_EXE = os.environ.get(
    "METASHAPE_EXE", r"C:\Program Files\Agisoft\Metashape\metashape.exe")

IMG_EXT = {".jpg", ".jpeg", ".tif", ".tiff", ".dng"}


def _stage(input_dir: Path, staging_root: Path) -> Path:
    """Link/copy input images into <flight>_unprocessed/photos/<flight>/."""
    images = [p for p in input_dir.rglob("*") if p.suffix.lower() in IMG_EXT]
    if not images:
        sys.exit(f"No drone images found in {input_dir}")
    name = input_dir.name or "flight"
    photos_dir = staging_root / f"{name}_unprocessed" / "photos" / name
    photos_dir.mkdir(parents=True, exist_ok=True)
    linked, copied = 0, 0
    for img in images:
        dest = photos_dir / img.name
        if dest.exists():
            continue
        try:
            os.link(img, dest)          # same-volume hardlink — instant
            linked += 1
        except OSError:
            shutil.copy2(img, dest)
            copied += 1
    log.info("Staged %d images (%d linked, %d copied)", len(images), linked, copied)
    return staging_root


def _write_config(staging_root: Path) -> Path:
    cfg = {
        "gpu_option": "0",
        "cpu_enabled": False,
        "input_folder": str(staging_root),
        "tmp_folder": str(staging_root / "_tmp_metashape"),
        "log_dir": str(staging_root / "log-files"),
        "gcp_csv": os.environ.get("AEROPOINTS_CSV", ""),
        "target_crs": os.environ.get("METASHAPE_CRS", "EPSG::28356"),
    }
    cfg_path = staging_root / "adapter_config.yaml"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg_path


def _band_count(path: Path) -> int:
    try:
        import rasterio
        with rasterio.open(path) as ds:
            return ds.count
    except Exception:
        return 0


def _collect_outputs(staging_root: Path, out_dir: Path) -> list[Path]:
    """Copy processed orthos into the pipeline output dir with canonical names."""
    written = []
    for processed in staging_root.glob("*_processed"):
        export = processed / "export"
        if not export.is_dir():
            continue
        for tif in sorted(export.glob("*_orthomosaic.tif")):
            n = _band_count(tif)
            if n >= 4:
                dest = out_dir / "orthophoto_ms.tif"
            else:
                dest = out_dir / "orthophoto.tif"
            shutil.copy2(tif, dest)
            log.info("Ortho (%d bands) → %s", n, dest)
            written.append(dest)

            if n >= 4 and NDVI_PY.exists():
                ndvi = out_dir / "ndvi.tif"
                r = subprocess.run(
                    [sys.executable, str(NDVI_PY), str(dest),
                     "--ndvi-out", str(ndvi)])
                if r.returncode == 0 and ndvi.exists():
                    written.append(ndvi)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True, help="Dropped imagery dir")
    ap.add_argument("--output", required=True, help="Pipeline output dir")
    args = ap.parse_args()

    input_dir = Path(args.input).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exe = Path(METASHAPE_EXE)
    if not exe.exists():
        sys.exit(f"Metashape not found at {exe} — set METASHAPE_EXE")

    with tempfile.TemporaryDirectory(prefix="ms_stage_") as tmp:
        staging = Path(tmp)
        _stage(input_dir, staging)
        cfg = _write_config(staging)
        cmd = [str(exe), "-r", str(MAIN_PY), str(cfg), "--yes"]
        log.info("Running: %s", " ".join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f"Metashape failed (exit {r.returncode})")
        written = _collect_outputs(staging, out_dir)

    if not written:
        sys.exit("Metashape produced no orthomosaic outputs")
    log.info("Done — %d output(s)", len(written))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
