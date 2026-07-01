
# 🌀 Buoyancy-Adaptive DOD Surrogate for Rod-Bundle Transients

This repository accompanies the manuscript:

**A Buoyancy-Adaptive Reduced-Order Surrogate for Heated Rod-Bundle Transients Using Gauge-Consistent Deep Orthogonal Decomposition**

In plain words: this project is about making CFD-informed rod-bundle thermal-hydraulic fields much faster to reconstruct, especially when heating and buoyancy start to mess with the transverse flow structures.

The source code will be released after the paper is published.

---

## What is this repo about?

Heated rod-bundle flow is not just “flow goes up, temperature goes up”.

When the flow rate drops or the heating becomes laterally skewed, buoyancy starts to matter. The transverse secondary flow may change its strength, move to another region, or even reorganize its topology. That is exactly the kind of behavior that a single fixed POD basis does not like.

So this work asks a simple question:

> Can we build a reduced-order surrogate that adapts its basis when buoyancy changes the flow structure?

The answer explored in the paper is: yes, by using a buoyancy-adaptive Deep Orthogonal Decomposition (DOD) basis with gauge-consistent coordinates.

---

## Why not just use POD?

POD is great when the dominant structures stay more or less in the same place.

For the axial velocity field, that is mostly fine.  
For the temperature field, it also works quite well in this case.  
But for the transverse velocity components, buoyancy can reshape the secondary flow. Then a fixed global basis has to work harder, because it is trying to describe moving or reorganizing structures using the same modes everywhere.

That is where DOD comes in.

Instead of using one fixed global basis, the DOD surrogate builds a local orthonormal basis that changes with the buoyancy state. In this work, the basis is driven by buoyancy-related descriptors, so the surrogate can adapt when the flow moves from forced convection toward mixed convection.

---

## What does the method do?

The surrogate reconstructs:

- transverse velocity component `Ux`;
- transverse velocity component `Uy`;
- axial velocity component `Uz`;
- temperature field `T`.

The workflow is roughly:

1. Generate steady CFD snapshots for heated rod-bundle operating points.
2. Compress the high-dimensional fields into reduced spaces.
3. Build a buoyancy-adaptive DOD basis for field reconstruction.
4. Align the local bases with a Procrustes-based gauge-consistency step.
5. Predict reduced coordinates using operating parameters.
6. Query the surrogate during heated flow transients.
7. Correct the temperature query using a residence-time-based thermal lag.
8. Use a quasi-steady hydraulic closure for the slowly varying flow ramps considered in the paper.

So the model is not trying to replace CFD in general. It is trying to carry useful CFD-resolved information into a fast surrogate that can be used in transient digital-twin-style calculations.

---

## What problem does it actually solve?

The main target is the buoyancy-sensitive transverse flow.

Compared with a global POD using the same coefficient mapper, the DOD surrogate improves the reconstruction of the transverse velocity components. The improvement is specific: it shows up where the flow structures reorganize with buoyancy.

This is also the main message of the paper:

> Adaptive bases are not automatically better for every field.  
> They are useful when the dominant structures move, rotate, split, or change topology with the operating condition.

In this rod-bundle case, that means the transverse secondary flow benefits most from the buoyancy-adaptive basis, while the axial velocity and temperature fields are already well handled by simpler bases.

---

## What will be included?

After publication, this repository will provide the implementation used in the paper, including:

- buoyancy-adaptive Deep Orthogonal Decomposition (DOD);
- gauge alignment and Procrustes-based basis-consistency procedures;
- global POD baseline;
- clustered-POD baseline;
- coefficient-regression comparison scripts;
- DOD/POD ablation studies;
- reconstruction-error analysis;
- buoyancy-band error decomposition;
- subspace/oracle-error comparison;
- modal-cost comparison;
- scripts for generating the main figures and tables.

At the moment, this repository is a public placeholder for the manuscript.

---

## What is not included?

The geometry, mesh, and CFD snapshot files are large, so they are not hosted directly in this GitHub repository.

These files may be made available upon reasonable request, subject to storage, transfer, and sharing constraints. Please contact the author through GitHub or by email if you need access to the large files required for full reproduction.

Third-party benchmark data, including the publicly documented PNL rod-bundle experimental references, remain subject to their original sources and citation requirements.

---

## Why this matters

Running CFD for every transient query is expensive.  
System codes are fast, but they do not resolve the three-dimensional local flow structures inside a rod bundle.  
A reduced-order surrogate sits somewhere in between.

The goal here is to let a system-level model keep its fast scalar closure path, while still giving access to local CFD-informed velocity and temperature fields for interpretation, monitoring, and digital-twin-style analysis.

Think of it as:

> Do not run CFD every time.  
> Do not throw away the CFD physics either.  
> Compress it, adapt it, and query it fast.

---

## Current status

The paper is currently under submission/preparation.

The code will be released after publication to keep the public repository consistent with the final accepted version of the manuscript.

---

## Citation

If you use this repository or build on this work, please cite the associated manuscript:

> X. Zhao, M. Peng, G. Chen, and X. Zhang,  
> **A Buoyancy-Adaptive Reduced-Order Surrogate for Heated Rod-Bundle Transients Using Gauge-Consistent Deep Orthogonal Decomposition**,  
> submitted manuscript.

The citation information will be updated after publication.

---
## appendix

During my research, countless open-source experimental data and code have inspired me. I will do my best to contribute to the development of the community and the development of reduced-order models. I am researching how to upload the geometric model to a cloud drive, but it will take some time.

## License

The released source code will be distributed under the MIT License.

Large CFD files, geometry files, mesh files, and third-party benchmark data are not covered by the source-code license and may be subject to separate sharing conditions.
