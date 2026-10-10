from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32, depth: int = 3):
        super().__init__()
        layers = []
        width = input_dim
        for _ in range(depth):
            layers.extend((nn.Linear(width, hidden_dim), nn.SiLU()))
            width = hidden_dim
        layers.append(nn.Linear(width, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class DODNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        ambient_dim: int,
        n_modes: int,
        hidden_dim: int = 32,
        depth: int = 3,
        delta_scale: float = 0.5,
    ):
        super().__init__()
        if not 1 <= n_modes <= ambient_dim:
            raise ValueError("n_modes must be between 1 and ambient_dim")
        self.ambient_dim = ambient_dim
        self.n_modes = n_modes
        self.delta_scale = float(delta_scale)
        self.basis = MLP(input_dim, ambient_dim * n_modes, hidden_dim, depth)
        nn.init.zeros_(self.basis.net[-1].weight)
        nn.init.zeros_(self.basis.net[-1].bias)
        reference = torch.zeros(ambient_dim, n_modes)
        reference[:n_modes] = torch.eye(n_modes)
        self.register_buffer("V0", reference)

    def perturbation(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.basis(inputs).reshape(-1, self.ambient_dim, self.n_modes) * self.delta_scale

    def raw_basis(self, inputs: torch.Tensor) -> torch.Tensor:
        candidate = self.V0.unsqueeze(0) + self.perturbation(inputs)
        q, _ = torch.linalg.qr(candidate, mode="reduced")
        return q

    def aligned_basis(self, inputs: torch.Tensor) -> torch.Tensor:
        q = self.raw_basis(inputs)
        cross = torch.einsum("bai,aj->bij", q, self.V0)
        u, _, vh = torch.linalg.svd(cross, full_matrices=False)
        rotation = u @ vh
        invalid = ~torch.isfinite(rotation).flatten(1).all(dim=1)
        if invalid.any():
            rotation = rotation.clone()
            rotation[invalid] = torch.eye(
                self.n_modes, dtype=rotation.dtype, device=rotation.device
            )
        return torch.einsum("bai,bij->baj", q, rotation)

    def project_reconstruct(
        self, inputs: torch.Tensor, coefficients: torch.Tensor, align: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        basis = self.aligned_basis(inputs) if align else self.raw_basis(inputs)
        local_coefficients = torch.einsum("ban,ba->bn", basis, coefficients)
        reconstructed = torch.einsum("ban,bn->ba", basis, local_coefficients)
        return reconstructed, local_coefficients, basis


def relative_projection_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = ((predicted - target) ** 2).sum(dim=1)
    denominator = (target**2).sum(dim=1).clamp_min(1e-12)
    return (numerator / denominator).mean()
