"""Run a preliminary five-state Global POD on the public VTU snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


RE_FILES = {
    500: "re500_60_60_1000.vtu",
    2000: "re2000_60_60_1000.vtu",
    3500: "re3500_60_60_1000.vtu",
    5000: "re5000_60_60_1000.vtu",
    6500: "re6500_60_60_1000.vtu",
}


def find_vector_name(data):
    names = []
    for name in data:
        values = np.asarray(data[name])
        if values.ndim == 2 and values.shape[1] == 3:
            names.append(name)
    preferred = [name for name in names if any(t in name.lower() for t in ("velocity", "vel", "u"))]
    return preferred[0] if preferred else (names[0] if len(names) == 1 else None)


def get_velocity(data):
    vector_name = find_vector_name(data)
    if all(name in data for name in ("u", "v", "w")):
        return ("u", "v", "w"), np.column_stack([
            np.asarray(data[name], dtype=np.float64) for name in ("u", "v", "w")
        ])
    if vector_name is None:
        return None, None
    return vector_name, np.asarray(data[vector_name], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-rank", type=int, default=5)
    args = parser.parse_args()

    try:
        import pyvista as pv
    except ImportError as exc:
        raise SystemExit("pyvista is required; install it with `python -m pip install pyvista`") from exc

    arrays = []
    field_name = None
    reference_points = None
    n_points = None
    for re_number, filename in RE_FILES.items():
        path = args.root / "public_CFD" / filename
        if not path.exists() or path.stat().st_size == 0:
            raise SystemExit(f"missing or incomplete file: {path}")
        grid = pv.read(path)
        current_name, velocity = get_velocity(grid.point_data)
        if current_name is None:
            raise SystemExit(f"no point-data vector field found in {path}")
        if field_name is None:
            field_name = current_name
            reference_points = np.asarray(grid.points)
            n_points = grid.n_points
        if current_name != field_name or grid.n_points != n_points:
            raise SystemExit(f"incompatible field layout in {path}: {current_name}, n_points={grid.n_points}")
        if not np.allclose(np.asarray(grid.points), reference_points, rtol=0.0, atol=1e-12):
            raise SystemExit(f"mesh point coordinates differ in {path}")
        arrays.append(velocity.reshape(-1))
        print(f"loaded Re={re_number}: {grid.n_points} points, field={field_name}")

    X = np.column_stack(arrays)
    valid = np.all(np.isfinite(X), axis=1)
    if not np.any(valid):
        raise SystemExit("no degree of freedom is finite in all five snapshots")
    X = X[valid]
    mean = X.mean(axis=1, keepdims=True)
    centered = X - mean
    U, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    rank = min(args.max_rank, len(singular_values))
    errors = {}
    for i, re_number in enumerate(RE_FILES):
        state = X[:, i]
        errors[str(re_number)] = {}
        norm = float(np.linalg.norm(state))
        for r in range(1, rank + 1):
            reconstruction = mean[:, 0] + U[:, :r] @ (U[:, :r].T @ (state - mean[:, 0]))
            errors[str(re_number)][str(r)] = float(np.linalg.norm(state - reconstruction) / norm)

    output = args.root / "POD" / "public_assessment"
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "singular_values.npy", singular_values)
    np.save(output / "snapshot_mean.npy", mean[:, 0])
    summary = {
        "reynolds": list(RE_FILES),
        "field": field_name,
        "n_points": n_points,
        "snapshot_matrix_shape": list(X.shape),
        "finite_common_dof_count": int(np.sum(valid)),
        "singular_values": singular_values.tolist(),
        "normalized_singular_energy": (singular_values**2 / np.sum(singular_values**2)).tolist(),
        "projection_error_by_re_and_rank": errors,
        "interpretation": "Five states only; preliminary feasibility diagnostic, not a final DOD dataset.",
    }
    (output / "pod_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
