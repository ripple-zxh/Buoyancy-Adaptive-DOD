"""Five-state POD/leave-one-out/local-subspace comparison.

This is a concept-validation script, not a neural DOD trainer. Every held-out
target is excluded from the training mean, POD basis, local weighted basis, and
interpolation endpoint fitting.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


RE = np.array([500, 2000, 3500, 5000, 6500], dtype=int)
FILES = {int(re): f"re{re}_60_60_1000.vtu" for re in RE}
REGIME = {500: "laminar", 2000: "transitional", 3500: "transitional/weakly turbulent", 5000: "turbulent", 6500: "turbulent"}
RANKS = (1, 2, 3)
LOCAL_BANDWIDTH = 1500.0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_snapshots(root: Path):
    try:
        import pyvista as pv
    except ImportError as exc:
        raise SystemExit("pyvista is required; install it with `python -m pip install pyvista`") from exc

    point_arrays = []
    coordinates = None
    raw_points = None
    for re in RE:
        path = root / "public_CFD" / FILES[int(re)]
        if not path.exists() or path.stat().st_size == 0:
            raise SystemExit(f"missing or incomplete file: {path}")
        grid = pv.read(path)
        if not all(name in grid.point_data for name in ("u", "v", "w")):
            raise SystemExit(f"expected scalar u/v/w point data in {path}")
        points = np.asarray(grid.points, dtype=np.float64)
        values = np.column_stack([
            np.asarray(grid.point_data[name], dtype=np.float64)
            for name in ("u", "v", "w")
        ])
        if coordinates is None:
            coordinates = points
            raw_points = int(grid.n_points)
        elif points.shape != coordinates.shape or not np.allclose(points, coordinates, rtol=0.0, atol=1e-12):
            raise SystemExit(f"mesh coordinates differ in Re={re}")
        point_arrays.append(values)
        print(f"loaded Re={int(re)}")

    values = np.stack(point_arrays, axis=0)  # state, point, component
    common_point_mask = np.all(np.isfinite(values), axis=(0, 2))
    values = values[:, common_point_mask, :]
    coordinates = coordinates[common_point_mask]
    n_points = values.shape[1]
    # q = [u_1...u_N, v_1...v_N, w_1...w_N], then columns are states.
    X = np.vstack([values[:, :, component].T for component in range(3)])
    del values
    return X, coordinates, raw_points, n_points


def pod_basis(X: np.ndarray, train: list[int], rank: int, *, anchor_re: float | None = None, bandwidth: float | None = None):
    params = RE[train].astype(float)
    if anchor_re is None or bandwidth is None:
        weights = np.ones(len(train), dtype=np.float64)
    else:
        weights = np.exp(-0.5 * ((params - anchor_re) / bandwidth) ** 2)
    weights /= weights.sum()
    mean = X[:, train] @ weights
    centered = (X[:, train] - mean[:, None]) * np.sqrt(weights)[None, :]
    gram = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    singular_values = np.sqrt(np.clip(eigenvalues[order], 0.0, None))
    usable = singular_values > max(float(singular_values[0]) * 1e-12, 1e-14)
    if rank > int(np.sum(usable)):
        raise ValueError(f"requested rank {rank} but only {int(np.sum(usable))} basis vectors are available")
    vectors = eigenvectors[:, order][:, usable]
    basis = centered @ vectors[:, :rank]
    basis /= singular_values[:rank][None, :]
    return mean, basis, singular_values, weights


def errors(q: np.ndarray, reconstruction: np.ndarray, n_points: int) -> dict[str, float]:
    residual = q - reconstruction
    result = {"total_error": float(np.linalg.norm(residual) / max(np.linalg.norm(q), 1e-30))}
    for index, name in enumerate(("u_error", "v_error", "w_error")):
        begin = index * n_points
        end = (index + 1) * n_points
        result[name] = float(np.linalg.norm(residual[begin:end]) / max(np.linalg.norm(q[begin:end]), 1e-30))
    return result


def project(q: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return mean + basis @ (basis.T @ (q - mean))


def geodesic_interpolate(phi0: np.ndarray, phi1: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate two equal-rank orthonormal subspaces on the Grassmann manifold."""
    left, singular, right_t = np.linalg.svd(phi0.T @ phi1, full_matrices=False)
    singular = np.clip(singular, -1.0, 1.0)
    angles = np.arccos(singular)
    principal0 = phi0 @ left
    principal1 = phi1 @ right_t.T
    tangent = np.zeros_like(principal1)
    for column, angle in enumerate(angles):
        sine = np.sin(angle)
        if sine > 1e-10:
            tangent[:, column] = (principal1[:, column] - singular[column] * principal0[:, column]) / sine
    interpolated = principal0 * np.cos(alpha * angles)[None, :] + tangent * np.sin(alpha * angles)[None, :]
    q, _ = np.linalg.qr(interpolated)
    return q[:, :phi0.shape[1]]


def topology_metrics(X: np.ndarray, coordinates: np.ndarray, n_points: int, index: int) -> dict:
    u = X[:n_points, index]
    v = X[n_points:2 * n_points, index]
    w = X[2 * n_points:, index]
    speed = np.sqrt(u * u + v * v + w * w)
    z = coordinates[:, 2]
    unique_z, inverse = np.unique(z, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_inverse)) + 1]
    ends = np.r_[starts[1:], len(order)]
    axial_min = np.full(len(unique_z), np.nan)
    axial_max = np.full(len(unique_z), np.nan)
    reverse_fraction = np.full(len(unique_z), np.nan)
    centerline = np.full(len(unique_z), np.nan)
    radius2 = coordinates[:, 0] ** 2 + coordinates[:, 1] ** 2
    for group, (start, end) in enumerate(zip(starts, ends)):
        ids = order[start:end]
        values = w[ids]
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        finite_values = values[finite]
        axial_min[group] = np.min(finite_values)
        axial_max[group] = np.max(finite_values)
        reverse_fraction[group] = np.mean(finite_values < 0.0)
        finite_ids = ids[finite]
        centerline[group] = w[finite_ids[np.argmin(radius2[finite_ids])]]

    recirc = np.flatnonzero(np.isfinite(axial_min) & (axial_min < 0.0))
    max_centerline_group = int(np.nanargmax(centerline))
    max_centerline = float(centerline[max_centerline_group])
    after_peak = np.arange(max_centerline_group + 1, len(unique_z))
    half_candidates = after_peak[np.isfinite(centerline[after_peak]) & (centerline[after_peak] <= 0.5 * max_centerline)]
    zero_candidates = after_peak[np.isfinite(centerline[after_peak]) & (centerline[after_peak] <= 0.0)]
    finite_w = np.isfinite(w)
    reverse = finite_w & (w < 0.0)
    return {
        "re": int(RE[index]),
        "regime": REGIME[int(RE[index])],
        "speed_min": float(np.nanmin(speed)),
        "speed_max": float(np.nanmax(speed)),
        "reverse_volume_proxy_fraction": float(np.mean(reverse[finite_w])),
        "reverse_speed_l1_fraction": float(np.sum(np.abs(w[reverse])) / max(np.sum(np.abs(w[finite_w])), 1e-30)),
        "u_z_min": float(np.nanmin(w)),
        "u_z_max": float(np.nanmax(w)),
        "recirculation_z_min": float(unique_z[recirc[0]]) if len(recirc) else None,
        "recirculation_z_max": float(unique_z[recirc[-1]]) if len(recirc) else None,
        "recirculation_length_proxy": float(unique_z[recirc[-1]] - unique_z[recirc[0]]) if len(recirc) else None,
        "jet_breakdown_proxy_z_half_centerline": float(unique_z[half_candidates[0]]) if len(half_candidates) else None,
        "centerline_zero_crossing_z": float(unique_z[zero_candidates[0]]) if len(zero_candidates) else None,
        "centerline_peak_z": float(unique_z[max_centerline_group]),
        "centerline_peak_w": max_centerline,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "POD" / "public_assessment"
    output.mkdir(parents=True, exist_ok=True)
    X, coordinates, raw_points, n_points = load_snapshots(root)

    preprocessing = {
        "reynolds": RE.tolist(),
        "raw_points_per_snapshot": raw_points,
        "common_finite_points": n_points,
        "state_definition": "q=[u_1..u_N, v_1..v_N, w_1..w_N]^T",
        "component_order": ["u", "v", "w"],
        "coordinate_order": ["x", "y", "z"],
        "axial_coordinate": "z",
        "centering": "training-set mean for each basis; target excluded in LOO and adaptive tests",
        "normalization": "none; raw velocity units retained; relative L2 errors are reported",
        "local_bandwidth_re": LOCAL_BANDWIDTH,
        "target_in_basis_assertion": False,
    }
    (output / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")

    n_state = X.shape[0]
    global_mean, global_u, global_s, _ = pod_basis(X, list(range(len(RE))), 4)
    global_rows = []
    for index, re in enumerate(RE):
        for rank in (1, 2, 3, 4):
            reconstruction = project(X[:, index], global_mean, global_u[:, :rank])
            row = {"re": int(re), "regime": REGIME[int(re)], "rank": rank, "basis": "global", "target_in_basis": True}
            row.update(errors(X[:, index], reconstruction, n_points))
            global_rows.append(row)
    write_csv(output / "global_pod_errors.csv", global_rows)

    loo_rows = []
    loo_means = {}
    for target, re in enumerate(RE):
        train = [i for i in range(len(RE)) if i != target]
        mean, basis3, _, _ = pod_basis(X, train, 3)
        loo_means[target] = mean
        for rank in RANKS:
            reconstruction = project(X[:, target], mean, basis3[:, :rank])
            row = {"re": int(re), "regime": REGIME[int(re)], "rank": rank, "basis": "loo_global", "target_in_basis": False, "train_re": ";".join(str(int(RE[i])) for i in train)}
            row.update(errors(X[:, target], reconstruction, n_points))
            loo_rows.append(row)
    write_csv(output / "loo_pod_errors.csv", loo_rows)

    adaptive_rows = []
    local_bases = []
    local_means = []
    local_trains = []
    for target, re in enumerate(RE):
        train = [i for i in range(len(RE)) if i != target]
        mean, basis3, _, weights = pod_basis(X, train, 3, anchor_re=float(re), bandwidth=LOCAL_BANDWIDTH)
        local_bases.append(basis3)
        local_means.append(mean)
        local_trains.append(train)

    for target, re in enumerate(RE):
        train = local_trains[target]
        mean = local_means[target]
        basis3 = local_bases[target]
        for rank in RANKS:
            reconstruction = project(X[:, target], mean, basis3[:, :rank])
            row = {"re": int(re), "regime": REGIME[int(re)], "method": "weighted_local_pod", "rank": rank, "target_in_basis": False, "train_re": ";".join(str(int(RE[i])) for i in train), "bandwidth_re": LOCAL_BANDWIDTH}
            row.update(errors(X[:, target], reconstruction, n_points))
            adaptive_rows.append(row)

        lower = [i for i in train if RE[i] < re]
        upper = [i for i in train if RE[i] > re]
        if lower and upper:
            lower_index = max(lower, key=lambda i: RE[i])
            upper_index = min(upper, key=lambda i: RE[i])
            alpha = float((re - RE[lower_index]) / (RE[upper_index] - RE[lower_index]))
            _, lower_basis, _, _ = pod_basis(X, train, 3, anchor_re=float(RE[lower_index]), bandwidth=LOCAL_BANDWIDTH)
            _, upper_basis, _, _ = pod_basis(X, train, 3, anchor_re=float(RE[upper_index]), bandwidth=LOCAL_BANDWIDTH)
            adaptive_basis = geodesic_interpolate(lower_basis, upper_basis, alpha)
            interpolation = f"grassmann:{int(RE[lower_index])}->{int(RE[upper_index])};alpha={alpha:.6g}"
        else:
            anchor = min(train, key=lambda i: abs(RE[i] - re))
            _, adaptive_basis, _, _ = pod_basis(X, train, 3, anchor_re=float(RE[anchor]), bandwidth=LOCAL_BANDWIDTH)
            interpolation = f"nearest_anchor_hold:{int(RE[anchor])}"
        for rank in RANKS:
            reconstruction = project(X[:, target], local_means[target], adaptive_basis[:, :rank])
            row = {"re": int(re), "regime": REGIME[int(re)], "method": "grassmann_interpolated", "rank": rank, "target_in_basis": False, "train_re": ";".join(str(int(RE[i])) for i in train), "bandwidth_re": LOCAL_BANDWIDTH, "interpolation": interpolation}
            row.update(errors(X[:, target], reconstruction, n_points))
            adaptive_rows.append(row)
    write_csv(output / "adaptive_basis_errors.csv", adaptive_rows)

    angle_rows = []
    for i in range(len(RE)):
        for j in range(i + 1, len(RE)):
            singular = np.linalg.svd(local_bases[i].T @ local_bases[j], compute_uv=False)
            angles = np.degrees(np.arccos(np.clip(singular, -1.0, 1.0)))
            row = {"re_i": int(RE[i]), "re_j": int(RE[j]), "adjacent": bool(j == i + 1)}
            for k, angle in enumerate(angles, start=1):
                row[f"principal_angle_{k}_deg"] = float(angle)
            row["mean_angle_deg"] = float(np.mean(angles))
            row["max_angle_deg"] = float(np.max(angles))
            angle_rows.append(row)
    write_csv(output / "principal_angles.csv", angle_rows)

    topology_rows = [topology_metrics(X, coordinates, n_points, index) for index in range(len(RE))]
    write_csv(output / "flow_topology_metrics.csv", topology_rows)

    # Keep the raw singular values for reproducibility and report the global energy split.
    np.save(output / "comparison_global_singular_values.npy", global_s)
    summary = {
        "global_centered_energy_fraction": (global_s ** 2 / np.sum(global_s ** 2)).tolist(),
        "state_length": n_state,
        "common_finite_points": n_points,
        "reynolds": RE.tolist(),
        "note": "Held-out target is excluded from LOO and adaptive means, bases, and interpolation endpoint fitting.",
    }
    (output / "adaptive_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # The evidence table is generated here so the markdown and CSVs cannot drift.
    global_by_key = {(int(row["re"]), int(row["rank"])): row for row in global_rows}
    loo_by_key = {(int(row["re"]), int(row["rank"])): row for row in loo_rows}
    adaptive_by_key = {(int(row["re"]), row["method"], int(row["rank"])): row for row in adaptive_rows}
    markdown = [
        "# FDA nozzle: five-state POD–adaptive subspace concept validation",
        "",
        "Status: completed from the five public velocity snapshots. This is a held-out concept-validation study, not a final deployable DOD model.",
        "",
        "## Data and leakage control",
        "",
        f"The state is `q=[u_1..u_N, v_1..v_N, w_1..w_N]` on {n_points:,} points that are finite in all five snapshots. No per-component rescaling was applied; all relative errors use the same raw velocity-unit definition. For every LOO/adaptive row, the target Reynolds number is excluded from the training mean, basis, local weighting, and interpolation endpoint fitting. `target_in_basis=False` is recorded in the CSV files.",
        "",
        "The weighted-local baseline uses a fixed Gaussian parameter bandwidth of Re=1500. It is intentionally a low-complexity proxy, not formal DOD. Grassmann interpolation uses the nearest lower/upper training anchors for interior targets and a nearest-anchor hold at the two endpoints; every anchor basis is refit using the same target-excluded training set.",
        "",
        "## Core comparison (total-state error)",
        "",
    ]
    for rank in RANKS:
        markdown += [f"### Rank {rank}", "", "| Re | regime | Global POD (in-sample) | LOO Global POD | Weighted Local POD | Grassmann adaptive | Local Δ vs LOO | Grassmann Δ vs LOO |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
        for re in RE:
            re_int = int(re)
            loo_value = loo_by_key[(re_int, rank)]["total_error"]
            local_value = adaptive_by_key[(re_int, "weighted_local_pod", rank)]["total_error"]
            grass_value = adaptive_by_key[(re_int, "grassmann_interpolated", rank)]["total_error"]
            improvement = float(loo_value) - float(local_value)
            grass_improvement = float(loo_value) - float(grass_value)
            global_value = float(global_by_key[(re_int, rank)]["total_error"])
            markdown.append(f"| {re_int} | {REGIME[re_int]} | {global_value:.6g} | {float(loo_value):.6g} | {float(local_value):.6g} | {float(grass_value):.6g} | {improvement:.6g} | {grass_improvement:.6g} |")
        markdown.append("")

    markdown += [
        "## Component-wise errors",
        "",
        "`global_pod_errors.csv`, `loo_pod_errors.csv`, and `adaptive_basis_errors.csv` contain separate `u_error`, `v_error`, and `w_error` columns in addition to `total_error`.",
        "",
        "## Subspace rotation and flow indicators",
        "",
        "`principal_angles.csv` contains pairwise principal angles of the rank-3 weighted-local subspaces. `flow_topology_metrics.csv` reports reverse-flow point-volume proxies, axial velocity extrema, and explicitly labeled centerline/recirculation proxies. These are not pressure-based metrics because the public VTU files contain no pressure field.",
        "",
        "## Final judgment",
        "",
        "**C — existing five snapshots are insufficient to decide whether adaptive representation generalizes.** The held-out comparison is useful and should guide the next CFD sweep, but five parameter values provide no independent transition-interval test set; rank-4 exact reconstruction is also algebraically guaranteed for five centered snapshots. The most valuable additions are dense Reynolds numbers around Re=1500–4000, especially 2000–3500, followed by a new train/test comparison.",
        "",
        "The current result should therefore not be reported as proof that DOD outperforms Global POD. It is a reproducible baseline and a leakage-controlled design for the next experiment.",
        "",
        "## Generated files",
        "",
        "- `global_pod_errors.csv`",
        "- `loo_pod_errors.csv`",
        "- `adaptive_basis_errors.csv`",
        "- `principal_angles.csv`",
        "- `flow_topology_metrics.csv`",
        "- `preprocessing.json`",
    ]
    (root / "docs" / "03_pod_adaptive_comparison.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
