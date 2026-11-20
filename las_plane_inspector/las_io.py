"""Read scaled LAS/LAZ coordinates with memory bounded by the sample and chunk sizes."""

from pathlib import Path

import laspy
import numpy as np


def sample_cloud(path: Path, limit: int, bounds, rng, chunk_size: int = 500_000):
    """Take a uniform sample without replacement from the selected spatial region."""
    points = np.empty((0, 3), dtype=np.float64)
    keys = np.empty(0, dtype=np.float64)
    eligible_count = 0
    read_count = 0

    with laspy.open(path) as reader:
        metadata = {
            "name": path.name,
            "las_version": str(reader.header.version),
            "point_format": reader.header.point_format.id,
            "source_point_count": reader.header.point_count,
            "coordinate_scale": reader.header.scales.tolist(),
        }
        for chunk in reader.chunk_iterator(chunk_size):
            read_count += len(chunk)
            coordinates = np.column_stack((chunk.x, chunk.y, chunk.z))
            if bounds is not None:
                low, high = bounds
                coordinates = coordinates[np.all((low <= coordinates) & (coordinates <= high), axis=1)]
            eligible_count += len(coordinates)
            # The lowest independent random keys select a uniform sample. Keys above
            # the largest retained key can never enter the final sample.
            chunk_keys = rng.random(len(coordinates))
            if len(keys) == limit:
                candidates = chunk_keys < keys.max()
                chunk_keys, coordinates = chunk_keys[candidates], coordinates[candidates]
            keys = np.concatenate((keys, chunk_keys))
            points = np.concatenate((points, coordinates))
            if len(keys) > limit:
                keep = np.argpartition(keys, limit - 1)[:limit]
                keys, points = keys[keep], points[keep]
    if read_count != metadata["source_point_count"]:
        raise ValueError("The LAS header point count does not match the data read.")
    metadata["eligible_point_count"] = eligible_count
    return points[np.argsort(keys)], metadata
