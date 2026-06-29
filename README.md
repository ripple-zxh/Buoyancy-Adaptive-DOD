```markdown
# Buoyancy-Adaptive DOD Surrogate for Transient Rod-Bundle Thermal-Hydraulics

This repository accompanies the manuscript:

**A Buoyancy-Adaptive Reduced-Order Surrogate for Heated Rod-Bundle Transients Using Gauge-Consistent Deep Orthogonal Decomposition**

The repository is associated with a reduced-order surrogate framework for heated rod-bundle transient thermal-hydraulics. The work focuses on how to reconstruct three-dimensional velocity and temperature fields during heated flow transients when buoyancy modifies the transverse secondary-flow structures.

---

## 1. Problem addressed

Heated rod-bundle transients involve coupled changes in flow rate, temperature, and buoyancy-driven secondary motion. Under reduced-flow or laterally skewed heating conditions, the flow can shift from nearly forced convection toward mixed convection. In this regime, the transverse velocity field does not only change in magnitude; its spatial structure, location, and topology may also reorganize with the operating condition.

This creates a difficulty for conventional reduced-order models based on a single fixed global basis, such as standard Proper Orthogonal Decomposition (POD). A global POD basis can efficiently represent nearly linear axial-flow responses, but it may become inefficient for buoyancy-sensitive transverse structures that move or reorganize across the parameter space.

The central problem addressed in this work is therefore:

> How can a reduced-order surrogate reconstruct velocity and temperature fields in heated rod-bundle transients while adapting to buoyancy-induced transverse-flow reorganization?

---

## 2. Method overview

The proposed framework combines a buoyancy-adaptive reduced-order field surrogate with thermal-hydraulic time-scale separation for transient deployment.

The main components are:

- **Buoyancy-adaptive Deep Orthogonal Decomposition (DOD)**  
  A parameter-dependent local orthonormal basis is constructed to adapt the reduced basis to the buoyancy state of the operating condition.

- **Gauge-consistent basis alignment**  
  Since local orthonormal bases are not unique up to rotations and reflections, a Procrustes-based gauge-alignment procedure is used to make the reduced coordinates consistent and suitable for coefficient regression.

- **POD and clustered-POD baselines**  
  Standard global POD and clustered POD are used as baseline methods to evaluate whether the adaptive basis provides a real advantage over fixed or piecewise fixed reduced bases.

- **Coefficient-regression comparisons**  
  Reduced coordinates are predicted from operating parameters, allowing the trained surrogate to reconstruct three-dimensional fields at unseen operating states.

- **Thermal-lag correction for transient queries**  
  A residence-time-based thermal-lag model is used to account for the finite transport time of coolant through the heated bundle. This enables a steady-trained temperature surrogate to be queried during heated transients.

- **Quasi-steady hydraulic closure**  
  For the smoothly varying flow ramps considered in the manuscript, the hydraulic response is treated quasi-steadily and coupled to the system-level response through a pressure-drop closure.

---

## 3. What the surrogate solves

The surrogate is designed to reconstruct the following fields for heated rod-bundle operating conditions:

- transverse velocity component \(U_x\);
- transverse velocity component \(U_y\);
- axial velocity component \(U_z\);
- temperature field \(T\).

Compared with a global POD using the same coefficient mapper, the buoyancy-adaptive DOD surrogate improves the reconstruction of the transverse velocity components, where buoyancy-induced flow reorganization is most important. The axial velocity and temperature fields are reconstructed with comparable accuracy by both bases, indicating that the adaptive basis is specifically useful for the nonlinear transverse-flow structures rather than being an indiscriminate improvement across all fields.

The repository will include scripts for reproducing the main analyses reported in the manuscript, including:

- end-to-end reconstruction-error comparison between DOD and POD;
- buoyancy-band error decomposition;
- subspace/oracle-error comparison;
- modal-cost comparison between DOD, global POD, and clustered POD;
- coefficient-regression ablation studies;
- post-processing scripts for the main figures and tables.

---

## 4. Scientific significance

This work is intended to support the development of real-time rod-bundle thermal-hydraulic digital twins. In such applications, a system-level solver requires fast scalar closure quantities, such as pressure drop, while local thermal-hydraulic interpretation requires access to three-dimensional velocity and temperature fields.

The framework separates these two requirements:

- the **system path** provides the hydraulic response needed for system-level momentum closure;
- the **field-reconstruction path** provides CFD-informed three-dimensional velocity and temperature fields for local-state awareness and physical interpretation.

This split-path structure allows high-fidelity rod-bundle thermal-hydraulic information to be carried into real-time-oriented transient analysis without forcing the system solver to evolve high-dimensional CFD fields directly.

The main methodological message is that a parameter-dependent reduced basis is most useful when the dominant flow structures change with the operating state. In the present problem, this occurs primarily in the buoyancy-sensitive transverse secondary flow.

---

## 5. Repository status

The source code will be released after publication of the paper.

The planned release will include:

- implementation of the buoyancy-adaptive DOD reduced-order model;
- gauge-alignment and Procrustes-based basis-consistency procedures;
- POD and clustered-POD baseline implementations;
- coefficient-regression comparison scripts;
- reconstruction-error and modal-cost analysis scripts;
- figure-generation and post-processing utilities for the main reproducibility results.

At the current stage, this repository serves as a public placeholder associated with the manuscript.

---

## 6. Data availability

The geometry, mesh, and CFD snapshot files used in the manuscript are large and are therefore not hosted directly in this GitHub repository.

After publication, the released code will provide the procedures needed to reproduce the reduced-order modeling workflow when the required data are available. Large geometry, mesh, and CFD snapshot files may be made available from the corresponding author upon reasonable request, subject to storage, transfer, and sharing constraints.

Third-party benchmark data, including publicly documented PNL rod-bundle experimental references, remain subject to their original data sources and citation requirements.

---

## 7. Citation

If you use this repository or build on this work, please cite the associated manuscript:

> X. Zhao, M. Peng, G. Chen, and X. Zhang,  
> **A Buoyancy-Adaptive Reduced-Order Surrogate for Heated Rod-Bundle Transients Using Gauge-Consistent Deep Orthogonal Decomposition**,  
> submitted manuscript.

The citation information will be updated after publication.

---

## 8. License

The released source code will be distributed under the MIT License.

Large CFD files, geometry files, mesh files, and third-party benchmark data are not covered by the source-code license and may be subject to separate sharing conditions.
```
