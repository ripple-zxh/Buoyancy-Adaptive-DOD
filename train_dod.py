from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor

from dod_core import DODNet, relative_projection_loss
from feature_transform import apply_standardizer, build_features, fit_standardizer


def _fit_ambient_pod(
    weighted_snapshots: torch.Tensor, ambient_dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = weighted_snapshots.mean(dim=0, keepdim=True)
    centered = weighted_snapshots - mean
    gram = centered @ centered.T
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = torch.flip(eigenvalues, dims=(0,)).clamp_min(0)
    eigenvectors = torch.flip(eigenvectors, dims=(1,))
    if eigenvalues.numel() == 0 or eigenvalues[0] <= 0:
        raise ValueError("snapshots have no nonzero centered variation")
    rank = int((eigenvalues > eigenvalues[0] * 1e-10).sum().item())
    used_dim = min(int(ambient_dim), rank)
    if used_dim < 1:
        raise ValueError("ambient POD rank is zero")
    singular_values = eigenvalues[:used_dim].sqrt().clamp_min(1e-8)
    basis = (centered.T @ eigenvectors[:, :used_dim]) / singular_values.unsqueeze(0)
    coefficients = centered @ basis
    return (
        mean.reshape(-1).cpu().numpy(),
        basis.cpu().numpy(),
        coefficients.cpu().numpy(),
    )


def _fit_basis(
    model: DODNet,
    basis_inputs: torch.Tensor,
    coefficients: torch.Tensor,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    gradient_clip: float,
) -> list[float]:
    optimizer = torch.optim.AdamW(
        model.basis.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_loss = float("inf")
    best_state = None
    wait = 0
    history = []
    train_ids = torch.as_tensor(train_indices, dtype=torch.long, device=basis_inputs.device)
    validation_ids = torch.as_tensor(validation_indices, dtype=torch.long, device=basis_inputs.device)
    for _ in range(max_epochs):
        optimizer.zero_grad(set_to_none=True)
        train_reconstruction = model.project_reconstruct(
            basis_inputs[train_ids], coefficients[train_ids], align=False
        )[0]
        train_loss = relative_projection_loss(train_reconstruction, coefficients[train_ids])
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.basis.parameters(), gradient_clip)
        optimizer.step()
        with torch.no_grad():
            validation_reconstruction = model.project_reconstruct(
                basis_inputs[validation_ids], coefficients[validation_ids], align=False
            )[0]
            validation_loss = float(
                relative_projection_loss(
                    validation_reconstruction, coefficients[validation_ids]
                ).item()
            )
        history.append(validation_loss)
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = {
                key: value.detach().clone()
                for key, value in model.basis.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.basis.load_state_dict(best_state)
    return history


def train_dod(
    parameters: np.ndarray,
    snapshots: np.ndarray,
    cell_volume: np.ndarray,
    n_modes: int,
    ambient_dim: int = 60,
    ri_median: float | None = None,
    validation_fraction: float = 0.2,
    seed: int = 0,
    hidden_dim: int = 32,
    depth: int = 3,
    delta_scale: float = 0.5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 1000,
    patience: int = 50,
    gradient_clip: float = 1.0,
    device: str | None = None,
) -> dict[str, Any]:
    parameter_array = np.asarray(parameters, dtype=np.float64)
    snapshot_array = np.asarray(snapshots, dtype=np.float32)
    volume_array = np.asarray(cell_volume, dtype=np.float64).reshape(-1)
    if parameter_array.ndim != 2 or parameter_array.shape[1] != 3:
        raise ValueError("parameters must have shape (n_samples, 3): [mdot, Ri, alpha]")
    if snapshot_array.ndim != 2 or snapshot_array.shape[0] != parameter_array.shape[0]:
        raise ValueError("snapshots must have shape (n_samples, n_cells)")
    if volume_array.size != snapshot_array.shape[1] or np.any(volume_array <= 0):
        raise ValueError("cell_volume must contain one positive value per snapshot cell")
    if not np.isfinite(parameter_array).all() or not np.isfinite(snapshot_array).all():
        raise ValueError("parameters and snapshots must contain only finite values")
    if len(parameter_array) < 5:
        raise ValueError("at least five snapshots are required for the internal validation split")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    if n_modes < 1 or ambient_dim < n_modes:
        raise ValueError("require 1 <= n_modes <= ambient_dim")

    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    sqrt_volume = np.sqrt(volume_array).astype(np.float32)
    weighted_snapshots = torch.as_tensor(
        snapshot_array, dtype=torch.float32, device=selected_device
    ) * torch.as_tensor(sqrt_volume, dtype=torch.float32, device=selected_device)[None, :]
    mean_field, ambient_basis, coefficients = _fit_ambient_pod(weighted_snapshots, ambient_dim)
    used_ambient_dim = ambient_basis.shape[1]
    used_modes = min(int(n_modes), used_ambient_dim)

    if ri_median is None:
        ri_median = float(np.median(parameter_array[:, 1]))
    basis_raw, mapper_raw = build_features(parameter_array, ri_median)
    basis_mean, basis_scale = fit_standardizer(basis_raw)
    mapper_mean, mapper_scale = fit_standardizer(mapper_raw)
    basis_standardized = apply_standardizer(basis_raw, basis_mean, basis_scale)
    mapper_standardized = apply_standardizer(mapper_raw, mapper_mean, mapper_scale)

    permutation = np.random.default_rng(seed).permutation(len(parameter_array))
    n_validation = max(2, int(round(len(parameter_array) * validation_fraction)))
    n_validation = min(n_validation, len(parameter_array) - 2)
    validation_indices = np.sort(permutation[:n_validation])
    train_indices = np.sort(permutation[n_validation:])

    model = DODNet(
        basis_standardized.shape[1], used_ambient_dim, used_modes,
        hidden_dim=hidden_dim, depth=depth, delta_scale=delta_scale
    ).to(selected_device)
    basis_tensor = torch.as_tensor(basis_standardized, dtype=torch.float32, device=selected_device)
    coefficient_tensor = torch.as_tensor(coefficients, dtype=torch.float32, device=selected_device)
    history = _fit_basis(
        model, basis_tensor, coefficient_tensor, train_indices, validation_indices,
        learning_rate, weight_decay, max_epochs, patience, gradient_clip
    )

    model.eval()
    with torch.no_grad():
        aligned_basis = model.aligned_basis(basis_tensor).cpu().numpy()
    aligned_coefficients = np.einsum("ban,ba->bn", aligned_basis, coefficients)
    coefficient_mean, coefficient_scale = fit_standardizer(aligned_coefficients)
    normalized_targets = (aligned_coefficients - coefficient_mean) / coefficient_scale
    mapper = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=0,
    )
    mapper.fit(mapper_standardized, normalized_targets)

    return {
        "model": model,
        "mapper": mapper,
        "n_samples": int(len(parameter_array)),
        "mu_field": mean_field.astype(np.float32),
        "U_A": ambient_basis.astype(np.float32),
        "sqrt_vol": sqrt_volume.astype(np.float32),
        "basis_mean": basis_mean.astype(np.float64),
        "basis_scale": basis_scale.astype(np.float64),
        "mapper_mean": mapper_mean.astype(np.float64),
        "mapper_scale": mapper_scale.astype(np.float64),
        "ri_median": float(ri_median),
        "coefficient_mean": coefficient_mean.astype(np.float64),
        "coefficient_scale": coefficient_scale.astype(np.float64),
        "ambient_dim": int(used_ambient_dim),
        "n_modes": int(used_modes),
        "hidden_dim": int(hidden_dim),
        "depth": int(depth),
        "delta_scale": float(delta_scale),
        "history": history,
        "n_samples": int(len(parameter_array)),
        "training_config": {
            "validation_fraction": float(validation_fraction),
            "seed": int(seed),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "max_epochs": int(max_epochs),
            "patience": int(patience),
            "gradient_clip": float(gradient_clip),
            "rf_estimators": 400,
            "rf_min_samples_leaf": 2,
        },
    }


def save_bundle(bundle: dict[str, Any], output_dir: str | Path, name: str) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target / f"{name}.npz",
        mu_field=bundle["mu_field"],
        U_A=bundle["U_A"],
        sqrt_vol=bundle["sqrt_vol"],
        basis_mean=bundle["basis_mean"],
        basis_scale=bundle["basis_scale"],
        mapper_mean=bundle["mapper_mean"],
        mapper_scale=bundle["mapper_scale"],
        ri_median=np.asarray(bundle["ri_median"]),
        coefficient_mean=bundle["coefficient_mean"],
        coefficient_scale=bundle["coefficient_scale"],
        V0=bundle["model"].V0.detach().cpu().numpy(),
    )
    torch.save(
        {
            "basis_state_dict": bundle["model"].basis.state_dict(),
            "ambient_dim": bundle["ambient_dim"],
            "n_modes": bundle["n_modes"],
            "hidden_dim": bundle["hidden_dim"],
            "depth": bundle["depth"],
            "delta_scale": bundle["delta_scale"],
            "training_config": bundle["training_config"],
        },
        target / f"{name}.pt",
    )
    with (target / f"{name}.pkl").open("wb") as stream:
        pickle.dump(bundle["mapper"], stream)
    metadata = {
        "schema_version": 1,
        "n_samples": bundle["n_samples"],
        "ambient_dim": bundle["ambient_dim"],
        "n_modes": bundle["n_modes"],
        "feature_inputs": ["mdot", "Ri", "alpha"],
        "basis_features": ["log10(max(Ri, 1e-30))", "alpha_eff"],
        "mapper_features": ["mdot", "log10(max(Ri, 1e-30))", "alpha_eff"],
        "weighting": "snapshot_weighted = snapshot * sqrt(cell_volume)",
        "inverse_weighting": "physical_snapshot = weighted_snapshot / sqrt(cell_volume)",
        "training_config": bundle["training_config"],
    }
    (target / f"{name}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
