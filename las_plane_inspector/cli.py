"""Command-line entry point for LAS Plane Inspector."""

import argparse
from pathlib import Path
from uuid import uuid4

from laspy.errors import LaspyException
import numpy as np

from .las_io import sample_cloud
from .planes import assign, fit_planes, measure
from .report import save_ply, save_preview, save_reports


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract dominant geometric planes from a LAS or LAZ point cloud."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--threshold", type=float, required=True,
                        help="maximum point-to-plane distance in input coordinate units")
    parser.add_argument("--planes", type=int, default=3, help="maximum planes to extract (1–8)")
    parser.add_argument("--max-points", type=int, default=250_000,
                        help="maximum number of points retained after cropping")
    parser.add_argument("--bbox", type=float, nargs=6, metavar="COORD",
                        help="crop bounds: xmin ymin zmin xmax ymax zmax")
    parser.add_argument("--trials", type=int, default=300, help="RANSAC trials per plane")
    parser.add_argument("--min-support", type=float, default=.03,
                        help="minimum support as a fraction of the fitting pool")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--azimuth", type=float, default=-151)
    parser.add_argument("--elevation", type=float, default=17)
    parser.add_argument("--export-ply", action="store_true", help="save all sampled points with plane IDs")
    parser.add_argument("--output", type=Path, default=Path("results"),
                        help="a new directory for the inspection results")
    config = parser.parse_args(argv)
    if not 0 < config.threshold <= np.finfo(float).max / 2:
        parser.error("--threshold must be positive and finite")
    if not 1 <= config.planes <= 8 or config.max_points < 20 or config.trials < 1 or config.seed < 0:
        parser.error("require 1–8 planes, at least 20 max points, positive trials and a nonnegative seed")
    if not 0 < config.min_support <= 1 or not (
        np.isfinite(config.azimuth) and -90 <= config.elevation <= 90
    ):
        parser.error("require 0 < --min-support <= 1, a finite azimuth and elevation between -90 and 90")
    bounds = None
    if config.bbox is not None:
        bounds = np.asarray(config.bbox).reshape(2, 3)
        if not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
            parser.error("--bbox requires finite minimum coordinates below the maximum coordinates")

    try:
        inspect(config, bounds)
    except (OSError, ValueError, LaspyException) as error:
        parser.exit(2, f"las-plane-inspect: {error}\n")
    return 0


def inspect(config, bounds):
    if config.output.exists():
        raise ValueError("Choose a new output directory to preserve existing results.")
    seed = np.random.SeedSequence(config.seed)
    sample_rng, split_rng, fit_rng = [np.random.default_rng(child) for child in seed.spawn(3)]
    points, metadata = sample_cloud(config.input, config.max_points, bounds, sample_rng)
    if len(points) < 20:
        raise ValueError(f"The selected region contains only {len(points)} points (20 required).")

    permutation = split_rng.permutation(len(points))
    split = int(len(points) * .75)
    fit_indices, validation_indices = permutation[:split], permutation[split:]
    origin = points[fit_indices].mean(axis=0)
    centered = points - origin
    fit = centered[fit_indices]
    validation = centered[validation_indices]
    models = fit_planes(fit, config.planes, config.threshold, config.trials, config.min_support, fit_rng)
    if not len(models):
        raise ValueError("No plane met the support requirement. Check the crop, threshold and trial count.")

    labels, distances, rows = measure(validation, models, config.threshold, origin)
    coverage = float(np.mean(labels > 0))
    report = {
        **metadata,
        "bbox": config.bbox,
        "seed": config.seed,
        "max_points": config.max_points,
        "sampled_points": len(points),
        "fit_points": len(fit),
        "holdout_points": len(validation),
        "requested_planes": config.planes,
        "planes_found": len(models),
        "ransac_trials_per_plane": config.trials,
        "minimum_fit_support": config.min_support,
        "distance_threshold": config.threshold,
        "holdout_coverage": coverage,
        "fixed_plane_threshold_sweep": [
            {"threshold": float(threshold), "coverage": float(np.mean(distances <= threshold))}
            for threshold in (config.threshold * .5, config.threshold, config.threshold * 2)
        ],
        "metric_units": "input coordinate units",
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    staging = config.output.parent / f".las-inspection-{uuid4().hex}"
    staging.mkdir()
    try:
        save_reports(staging, rows, report)
        save_preview(staging / "preview.png", validation, labels, rows, report,
                     config.azimuth, config.elevation)
        if config.export_ply:
            sample_labels, _ = assign(centered, models, config.threshold)
            save_ply(staging / "segmented.ply", points, sample_labels)
        staging.rename(config.output)
    finally:
        if staging.exists():
            for name in ("planes.csv", "report.json", "preview.png", "segmented.ply"):
                (staging / name).unlink(missing_ok=True)
            staging.rmdir()
    print(f'{len(models)} plane(s) | {len(points):,} sampled points | '
          f'{len(validation):,} held out | {coverage:.1%} covered')
    print(f"Results: {config.output}")
