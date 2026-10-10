from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from dod_core import DODNet
from feature_transform import apply_standardizer, build_features


class DODInference:
    def __init__(self, bundle_dir: str | Path, name: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        root = Path(bundle_dir)
        with np.load(root / f"{name}.npz") as values:
            self.mu_field = values["mu_field"].astype(np.float64)
            self.U_A = values["U_A"].astype(np.float64)
            self.sqrt_vol = values["sqrt_vol"].astype(np.float64)
            self.basis_mean = values["basis_mean"]
            self.basis_scale = values["basis_scale"]
            self.mapper_mean = values["mapper_mean"]
            self.mapper_scale = values["mapper_scale"]
            self.ri_median = float(values["ri_median"])
            self.coefficient_mean = values["coefficient_mean"]
            self.coefficient_scale = values["coefficient_scale"]
            reference = values["V0"]
        try:
            checkpoint = torch.load(root / f"{name}.pt", map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(root / f"{name}.pt", map_location=self.device)
        self.model = DODNet(
            input_dim=len(self.basis_mean),
            ambient_dim=int(checkpoint["ambient_dim"]),
            n_modes=int(checkpoint["n_modes"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            depth=int(checkpoint["depth"]),
            delta_scale=float(checkpoint["delta_scale"]),
        ).to(self.device)
        self.model.V0.copy_(torch.as_tensor(reference, dtype=self.model.V0.dtype, device=self.device))
        self.model.basis.load_state_dict(checkpoint["basis_state_dict"])
        self.model.eval()
        with (root / f"{name}.pkl").open("rb") as stream:
            self.mapper = pickle.load(stream)

    def predict(self, parameters: np.ndarray) -> np.ndarray:
        values = np.asarray(parameters, dtype=np.float64)
        was_vector = values.ndim == 1
        if was_vector:
            values = values.reshape(1, -1)
        basis_raw, mapper_raw = build_features(values, self.ri_median)
        basis_inputs = apply_standardizer(basis_raw, self.basis_mean, self.basis_scale)
        mapper_inputs = apply_standardizer(mapper_raw, self.mapper_mean, self.mapper_scale)
        basis_tensor = torch.as_tensor(basis_inputs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            aligned_basis = self.model.aligned_basis(basis_tensor).cpu().numpy()
        coefficients = np.asarray(self.mapper.predict(mapper_inputs))
        if coefficients.ndim == 1:
            coefficients = coefficients.reshape(len(values), -1)
        coefficients = coefficients * self.coefficient_scale + self.coefficient_mean
        ambient_coefficients = np.einsum("ban,bn->ba", aligned_basis, coefficients)
        weighted_fields = self.mu_field[None, :] + ambient_coefficients @ self.U_A.T
        fields = weighted_fields / self.sqrt_vol[None, :]
        return fields[0] if was_vector else fields
