# 🌀 Buoyancy-Adaptive DOD Surrogate for Rod-Bundle Transients

This repository accompanies the research article:

**A Buoyancy-Adaptive Reduced-Order Surrogate for Heated Rod-Bundle Transients Using Gauge-Consistent Deep Orthogonal Decomposition**

Xinheng Zhao, Minjun Peng, Guangliang Chen, and Xiang Zhang

*Annals of Nuclear Energy*, **242**, 112896 (2027).

**DOI:** https://doi.org/10.1016/j.anucene.2026.112896

This project develops a reduced-order surrogate for fast reconstruction of thermal-hydraulic fields in heated rod bundles, with particular attention to transverse secondary-flow structures that change as buoyancy becomes important.

In simple terms, the central question is:

> Can the reduced basis itself adapt when buoyancy changes the flow structure?

The method uses a buoyancy-adaptive Deep Orthogonal Decomposition (DOD) basis together with gauge-consistent coordinates to construct a parameter-dependent reduced representation.

---

## What is included?

This repository provides the source-code implementation of the core DOD methodology developed in the paper.

The released code covers the core workflow from prepared CFD snapshots to DOD training, model serialization, inference, and physical-field reconstruction, including:

- Ambient POD compression of high-dimensional CFD fields;
- Construction of parameter-dependent local orthonormal bases;
- Buoyancy-aware feature construction;
- Procrustes-based gauge alignment;
- Construction of gauge-consistent reduced coordinates;
- Reduced-coordinate regression;
- DOD model training and serialization;
- Model loading and physical-field reconstruction at new operating conditions.

---

## Why not just use POD?

A global POD model uses a fixed basis.

When the dominant spatial structures remain approximately fixed as the operating parameters change, this can work very well.

The transverse secondary flow in a heated rod bundle can behave differently. As the flow and heating conditions change, buoyancy becomes more or less important, and the transverse structures may shift, rotate, strengthen, weaken, or reorganize.

A fixed global POD basis must represent all of these states using the same set of modes.

The basic idea of this work is therefore:

> Instead of requiring all operating conditions to share the same low-dimensional basis, allow the basis itself to adapt to the buoyancy state.

---

## Core method

The implementation uses a two-level reduced representation.

First, the volume-weighted CFD snapshots are compressed into a larger ambient POD space.

DOD then learns a lower-dimensional, parameter-dependent local subspace inside this ambient space.

For a given parameter state $\boldsymbol{\mu}$, a neural network predicts a perturbation of a reference basis $V_0$:

$$
V_0 + \Delta V(\boldsymbol{\mu})
$$

A reduced QR decomposition is then used to construct an orthonormal local basis:

$$
V_{\mathrm{raw}}(\boldsymbol{\mu})
=
\operatorname{qf}
\left[
V_0+\Delta V(\boldsymbol{\mu})
\right]
$$

Unlike a fixed POD basis, the resulting DOD basis therefore changes with the operating condition.

The local basis is trained by minimizing the relative projection error of the ambient POD coefficients.

---

## Gauge-consistent coordinates

A parameter-dependent orthonormal basis introduces an important ambiguity:

The same reduced subspace can be represented by different orthonormal bases.

Even when two local bases represent the same or very similar subspaces, their individual modes may rotate, change sign, or mix with one another.

This makes reduced coordinates from different operating conditions difficult to compare and regress consistently.

The implementation therefore applies an SVD-based orthogonal Procrustes alignment between each local basis and a common reference basis.

For the raw local basis $V_{\mathrm{raw}}(\boldsymbol{\mu})$, the alignment is obtained from

$$
V_{\mathrm{raw}}(\boldsymbol{\mu})^T V_0
=
U\Sigma W^T
$$

with the orthogonal rotation

$$
R(\boldsymbol{\mu})=UW^T.
$$

The aligned basis is

$$
V(\boldsymbol{\mu})
=
V_{\mathrm{raw}}(\boldsymbol{\mu})R(\boldsymbol{\mu}).
$$

This establishes a common gauge across the parameter space.

Reduced coordinates are then computed using the aligned local bases, producing gauge-consistent coordinates for subsequent parameter-to-coordinate regression.

---

## Buoyancy-aware features

The operating condition is represented by

$$
\boldsymbol{\mu}
=
(\dot{m},Ri,\alpha)
$$

where:

- $\dot{m}$ is the mass-flow-related operating parameter;
- $Ri$ characterizes the relative importance of buoyancy;
- $\alpha$ represents the lateral heating-asymmetry parameter used in this study.

For basis adaptation, the implementation uses

$$
\log_{10}(Ri)
$$

together with an effective heating-asymmetry feature:

$$
\alpha_{\mathrm{eff}}
=
\alpha
\frac{Ri}{Ri+Ri_{\mathrm{median}}}.
$$

As $Ri$ becomes small,

$$
\alpha_{\mathrm{eff}}\rightarrow 0,
$$

so the influence of lateral heating asymmetry on basis adaptation is naturally suppressed in the low-buoyancy regime.

As buoyancy becomes more important, the influence of $\alpha$ on the local basis is progressively restored.

The DOD basis therefore uses

$$
\left(
\log_{10}(Ri),
\alpha_{\mathrm{eff}}
\right)
$$

as its input features, while the coefficient mapper uses

$$
\left(
\dot{m},
\log_{10}(Ri),
\alpha_{\mathrm{eff}}
\right)
$$

to predict the gauge-consistent reduced coordinates.

---

## Training workflow

The core training procedure is:

1. Apply cell-volume weighting to the CFD snapshots.
2. Compute an ambient POD representation.
3. Convert the high-dimensional CFD fields into ambient POD coefficients.
4. Construct buoyancy-aware features from $(\dot{m},Ri,\alpha)$.
5. Standardize the input features.
6. Train the parameter-dependent DOD basis.
7. Enforce local basis orthonormality using reduced QR decomposition.
8. Optimize the local subspaces using the relative projection error.
9. Align the trained local bases to the reference gauge.
10. Compute gauge-consistent coordinates in the aligned local bases.
11. Train a regression mapper from the operating parameters to the reduced coordinates.
12. Save the DOD basis network, coordinate mapper, ambient POD representation, and reconstruction parameters.

The current implementation uses:

- An MLP for parameter-dependent basis adaptation;
- Reduced QR decomposition for basis orthogonalization;
- SVD-based orthogonal Procrustes alignment for gauge consistency;
- Random Forest regression for reduced-coordinate prediction.

---

## Inference and field reconstruction

For a new operating condition $(\dot{m},Ri,\alpha)$, the inference pipeline is:

```text
Operating parameters
        |
        v
Buoyancy-aware features
        |
        v
Parameter-dependent DOD basis
        |
        v
Gauge alignment
        |
        v
Reduced-coordinate prediction
        |
        v
Ambient POD coefficients
        |
        v
Weighted field reconstruction
        |
        v
Physical CFD-informed field
```

Once the offline CFD data have been generated and the DOD surrogate has been trained, spatial fields at new operating conditions can be reconstructed without rerunning CFD.

---

## Code structure

The core DOD implementation consists of four main files.

### `dod_core.py`

Implements the core DOD formulation, including:

- MLP basis network;
- Parameter-dependent basis perturbation;
- Reduced QR orthogonalization;
- Reference basis;
- Procrustes gauge alignment;
- Local projection;
- Reduced reconstruction;
- Relative projection loss.

### `feature_transform.py`

Implements the parameter-feature transformations, including:

- $\log_{10}(Ri)$;
- $\alpha_{\mathrm{eff}}$;
- Basis input features;
- Coefficient-mapper input features;
- Feature standardization.

### `train_dod.py`

Implements the training workflow, including:

- Cell-volume weighting;
- Ambient POD;
- Ambient coefficient extraction;
- Training/validation split;
- DOD basis optimization;
- Early stopping;
- Gradient clipping;
- Gauge-consistent coordinate construction;
- Random Forest coefficient mapping;
- Serialization of the trained model and reconstruction parameters.

### `dod_infer.py`

Implements model loading and inference, including:

- Loading the ambient POD representation;
- Loading the DOD basis network;
- Loading the coordinate mapper;
- Feature construction for new operating conditions;
- Aligned local-basis evaluation;
- Reduced-coordinate prediction;
- Ambient coefficient reconstruction;
- CFD-informed physical-field reconstruction.

Together, these four files constitute the core source-code implementation of the DOD methodology developed in this work.

---

## Data availability

The geometry, mesh, and CFD snapshot files are large and are therefore not hosted directly in this GitHub repository.

These files may be made available upon reasonable request, subject to storage, transfer, and sharing constraints. Please contact the author through GitHub or by email if access to the large files is required for reproduction.

Third-party benchmark data, including the publicly documented PNL rod-bundle experimental references, remain subject to their original sources and citation requirements.

---

## Citation

If you use this repository or build on this work, please cite the associated journal article:

**Zhao, X., Peng, M., Chen, G., & Zhang, X. (2027).**
A buoyancy-adaptive reduced-order surrogate for heated rod-bundle transients using gauge-consistent deep orthogonal decomposition.
*Annals of Nuclear Energy*, **242**, 112896.

https://doi.org/10.1016/j.anucene.2026.112896

### BibTeX

```bibtex
@article{ZHAO2027112896,
  title = {A buoyancy-adaptive reduced-order surrogate for heated rod-bundle transients using gauge-consistent deep orthogonal decomposition},
  journal = {Annals of Nuclear Energy},
  volume = {242},
  pages = {112896},
  year = {2027},
  issn = {0306-4549},
  doi = {10.1016/j.anucene.2026.112896},
  url = {https://www.sciencedirect.com/science/article/pii/S030645492600784X},
  author = {Xinheng Zhao and Minjun Peng and Guangliang Chen and Xiang Zhang},
  keywords = {Reduced-order modeling, Deep orthogonal decomposition, Digital twin, Multiscale coupling, Rod-bundle thermal-hydraulics}
}
```

---

## License

The source code is distributed under the MIT License.

Large CFD files, geometry files, mesh files, and third-party benchmark data are not covered by the source-code license and may be subject to separate sharing conditions.

---

## Appendix

During this research, many publicly available experimental datasets, numerical methods, and open-source scientific software projects provided valuable references and inspiration.

By releasing the core implementation of the DOD methodology developed in this work, we hope to contribute in turn to the thermal-hydraulics and reduced-order modeling communities.
