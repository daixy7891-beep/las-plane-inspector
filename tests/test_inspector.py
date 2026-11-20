import csv
import json
from pathlib import Path
import tempfile
import unittest

import laspy
import numpy as np

from las_plane_inspector.cli import main
from las_plane_inspector.las_io import sample_cloud
from las_plane_inspector.planes import assign


class InspectorTests(unittest.TestCase):
    def test_uniform_sample_is_independent_of_read_chunk_size(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "crop.las"
            las = laspy.create(point_format=0, file_version="1.2")
            las.x = np.arange(60, dtype=float)
            las.y = np.zeros(60)
            las.z = np.zeros(60)
            las.write(path)
            bounds = np.array([[10, -1, -1], [49, 1, 1]])
            a, meta = sample_cloud(path, 12, bounds, np.random.default_rng(9), chunk_size=7)
            b, _ = sample_cloud(path, 12, bounds, np.random.default_rng(9), chunk_size=49)
            self.assertEqual(meta["eligible_point_count"], 40)
            self.assertEqual(len(np.unique(a[:, 0])), 12)
            self.assertTrue(np.all((a[:, 0] >= 10) & (a[:, 0] <= 49)))
            np.testing.assert_array_equal(a, b)

    def test_assignment_uses_nearest_plane_and_leaves_outliers_unassigned(self):
        models = np.array([[0, 0, 1, 0], [1, 0, 0, 0]])
        points = np.array([[.02, 0, .04], [.03, 0, .01], [1, 1, 1]])
        labels, distance = assign(points, models, .05)
        np.testing.assert_array_equal(labels, [2, 1, 0])
        np.testing.assert_allclose(distance, [.02, .01, 1.0])

    def test_end_to_end_las_exports_plane_equations_at_global_coordinates(self):
        rng = np.random.default_rng(4)
        xy = rng.uniform(1, 5, (550, 2))
        yz = rng.uniform(1, 5, (400, 2))
        points = np.vstack((np.c_[xy, np.zeros(len(xy))], np.c_[np.zeros(len(yz)), yz]))
        points += [250_000, 3_400_000, 250]
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "scene.las"
            output = Path(folder) / "reports"
            las = laspy.create(point_format=6, file_version="1.4")
            las.header.offsets = [250_000, 3_400_000, 250]
            las.x, las.y, las.z = points.T
            las.write(source)
            result = main([str(source), "--threshold", ".04", "--planes", "2",
                           "--output", str(output), "--export-ply"])
            self.assertEqual(result, 0)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["fit_points"] + report["holdout_points"], len(points))
            self.assertEqual(report["holdout_coverage"], 1)
            with (output / "planes.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["orientation"] for row in rows], ["near-horizontal", "near-vertical"])
            self.assertAlmostEqual(float(rows[0]["d"]), -250, places=7)
            self.assertAlmostEqual(float(rows[1]["d"]), -250_000, places=7)
            self.assertTrue((output / "preview.png").is_file())
            with (output / "segmented.ply").open("rb") as stream:
                self.assertTrue(stream.readline().startswith(b"ply"))
