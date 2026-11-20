"""Fit normalized plane equations and evaluate them on a held-out point sample."""

import numpy as np


def fit_plane(points, threshold, trials, rng):
    """Find a three-point RANSAC model, then refine its support by orthogonal least squares."""
    best_mask = None
    best_score = (0, -np.inf)
    for _ in range(trials):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        edge_b, edge_c = b - a, c - a
        normal = np.cross(edge_b, edge_c)
        length = np.linalg.norm(normal)
        if length <= 1e-10 * np.linalg.norm(edge_b) * np.linalg.norm(edge_c):
            continue
        normal /= length
        distance = np.abs(points @ normal - (a @ normal))
        mask = distance <= threshold
        score = (int(mask.sum()), -float(np.sum(distance[mask] ** 2)))
        if score > best_score:
            best_score, best_mask = score, mask
    if best_mask is None or best_mask.sum() < 3:
        return None

    inliers = points[best_mask]
    center = inliers.mean(axis=0)
    centered = inliers - center
    values, vectors = np.linalg.eigh(centered.T @ centered)
    if values[1] - values[0] <= 1e-12 * values[-1]:
        return None
    normal = vectors[:, 0]
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    return np.r_[normal, -(center @ normal)]


def fit_planes(fit, maximum, threshold, trials, min_support, rng):
    """Extract up to the requested number of planes by peeling off inliers."""
    planes = []
    remaining = fit
    required = max(3, int(np.ceil(min_support * len(fit))))
    for _ in range(maximum):
        if len(remaining) < required:
            break
        sample = remaining[rng.choice(len(remaining), min(60_000, len(remaining)), replace=False)]
        model = fit_plane(sample, threshold, trials, rng)
        if model is None:
            break
        mask = np.abs(remaining @ model[:3] + model[3]) <= threshold
        if np.count_nonzero(mask) < required:
            break
        planes.append(model)
        remaining = remaining[~mask]
    return np.asarray(planes)


def assign(points, models, threshold):
    """Give each point its closest accepted plane ID, using 0 for unassigned points."""
    nearest = np.zeros(len(points), dtype=np.int32)
    distance = np.full(len(points), np.inf)
    for plane_id, model in enumerate(models, 1):
        residual = np.abs(points @ model[:3] + model[3])
        closer = residual < distance
        nearest[closer] = plane_id
        distance[closer] = residual[closer]
    return np.where(distance <= threshold, nearest, 0), distance


def measure(validation, models, threshold, origin):
    labels, distances = assign(validation, models, threshold)
    rows = []
    for plane_id, model in enumerate(models, 1):
        a, b, c, d = model
        residuals = distances[labels == plane_id]
        inclination = float(np.degrees(np.arccos(np.clip(abs(c), 0.0, 1.0))))
        orientation = "inclined"
        if inclination <= 15:
            orientation = "near-horizontal"
        elif inclination >= 75:
            orientation = "near-vertical"
        rows.append({
            "plane_id": plane_id,
            "orientation": orientation,
            "angle_to_xy_deg": inclination,
            "validation_inliers": len(residuals),
            "validation_share": len(residuals) / len(validation),
            "inlier_rmse": float(np.sqrt(np.mean(residuals ** 2))) if len(residuals) else None,
            "inlier_p95": float(np.quantile(residuals, .95)) if len(residuals) else None,
            "a": float(a), "b": float(b), "c": float(c),
            "d": float(d - model[:3] @ origin),
        })
    return labels, distances, rows
