"""Write the inspection report and a CloudCompare-compatible colored point sample."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np


COLORS = ["#62748A", "#F5AA55", "#61CCE2", "#72D39D", "#D89CF2",
          "#F78580", "#F0D975", "#A8B8FC", "#EF9DBF"]


def save_preview(path: Path, points, labels, rows, report, azimuth, elevation):
    ink = "#14243B"
    muted = "#4D6177"
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), facecolor="#F5F8FC")
    fig.subplots_adjust(left=.052, right=.948, bottom=.245, top=.785, wspace=.045)
    fig.text(.052, .924, "Dominant planes in a LAS point cloud", size=23, weight="bold", color=ink)
    fig.text(.053, .866,
             f'{report["source_point_count"]:,} source points  /  '
             f'{report["sampled_points"]:,} sampled  /  '
             f'{len(points):,} held out for evaluation', size=11, color=muted)

    az, el = np.radians([azimuth, elevation])
    right = np.array([-np.sin(az), np.cos(az), 0])
    up = np.array([-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el)])
    projected = points @ np.stack((right, up, np.cross(right, up))).T
    order = np.argsort(projected[:, 2])
    projected = projected[order]
    for ax, title, color in zip(
        axes, ("Input sample", "Plane assignments"), ("#C6D0DF", np.array(COLORS)[labels[order]])
    ):
        ax.set_title(title, color=ink, loc="left", fontsize=12, pad=9)
        ax.set_facecolor("#101D30")
        ax.scatter(projected[:, 0], projected[:, 1], c=color, s=.8, alpha=.75, linewidths=0)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        for edge in ax.spines.values():
            edge.set_visible(False)

    for row_index, row in enumerate(rows):
        x = .053 + (row_index % 4) * .223
        y = .177 - (row_index // 4) * .056
        fig.text(x, y, "●", size=10, color=COLORS[row["plane_id"]])
        fig.text(x + .018, y, f'{row["plane_id"]}  {row["orientation"]}', size=9, color=ink)
    fig.text(.053, .054,
             f'{report["holdout_coverage"]:.1%} holdout coverage at '
             f'{report["distance_threshold"]:g} input units  /  gray = unassigned',
             size=10, color=muted)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_ply(path: Path, points, labels):
    rgb = np.rint(np.array([to_rgb(color) for color in COLORS]) * 255).astype(np.uint8)
    vertices = np.empty(len(points), dtype=[
        ("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("red", "u1"), ("green", "u1"),
        ("blue", "u1"), ("plane_id", "<i4"),
    ])
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = rgb[labels].T
    vertices["plane_id"] = labels
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property double x\nproperty double y\nproperty double z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property int plane_id\nend_header\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        vertices.tofile(stream)


def save_reports(output, rows, report):
    with (output / "planes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
