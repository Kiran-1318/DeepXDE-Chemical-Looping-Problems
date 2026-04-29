# DeepXDE Chemical Looping Problems

A series of 10 original Physics-Informed Neural Network (PINN) problems
designed as a deliberate progression toward a PINN for Fe₂O₃ redox
kinetics in chemical looping gasification for hydrogen production.

All problems are **original formulations** — not tutorial replications.
Each problem teaches one specific skill needed for the core Fe₂O₃ project.
Problems 1–3 include both DeepXDE (TensorFlow) and raw PyTorch implementations.

## Repository structure
DeepXDE-Chemical-Looping-Problems/
├── 1_Batch_Reactor_First_Order_ODE.ipynb
├── 2_Arrhenius_Rate_Constant_ODE.ipynb
├── 3_NonIsothermal_Coupled_Reactor.ipynb
├── 4_Shrinking_Core_Model.ipynb
├── 5_Grain_Model_Comparison.ipynb
├── 6_Shrinking_Core_Arrhenius_TwoInput.ipynb
├── 7_Inverse_PINN_Rate_Constant.ipynb
├── 8_Inverse_PINN_Arrhenius_Parameters.ipynb
├── 9_MultiCycle_Redox_Kinetics.ipynb
└── 10_PhysicsData_Hybrid_PINN.ipynb
## Problem series

### Tier 1 — ODE kinetics (Problems 1–3)
Builds the mathematical skeleton for solid-gas kinetic modelling.

| # | Problem | Physics encoded | Key skill | Result |
|---|---------|----------------|-----------|--------|
| 1 | Isothermal first-order batch reactor | `dC/dt = -k·C` | ODE as physics residual, Adam + L-BFGS | Validated vs analytical `C₀·exp(-kt)` |
| 2 | Arrhenius temperature-dependent rate | `k(T) = A·exp(-Ea/RT)` | Encoding Arrhenius inside loss function | Validated analytically at multiple T |
| 3 | Non-isothermal coupled reactor | `dC/dt` + `dT/dt` coupled system | Coupled ODEs, loss weight balancing | DeepXDE + PyTorch both validated vs SciPy RK45 |

### Tier 2 — Solid-gas conversion (Problems 4–6)
Introduces the shrinking core model — the governing equation of the Fe₂O₃ PINN.

| # | Problem | Physics encoded | Key skill | Result |
|---|---------|----------------|-----------|--------|
| 4 | Shrinking core model | `dX/dt = k·(1-X)^(2/3)` | (1-X)^(2/3) nonlinearity in loss | Validated vs analytical solution |
| 5 | Grain model comparison | `dX/dt = k·(1-X)^(1/3)` | Three-model comparison on same axes | Shrinking core vs grain vs 1st order comparison |
| 6 | Shrinking core + Arrhenius, two-input network | `dX/dt = A·exp(-Ea/RT)·(1-X)^(2/3)` | Multi-input network, T as parameter input | Validated at 4 temperatures (600–900 K) |

### Tier 3 — Inverse problems (Problems 7–8)
Identifies unknown kinetic parameters from noisy synthetic data.

| # | Problem | What is identified | Key skill | Result |
|---|---------|-------------------|-----------|--------|
| 7 | Inverse PINN: recover rate constant k | k from 20 noisy observations | `dde.Variable`, trainable parameters | **k error: 0.13%** |
| 8 | Inverse PINN: recover A and Ea | Both A and Ea simultaneously from multi-temperature TGA data | Log-scale variables, loss weight balancing, per-temperature t_max | **A error: 3.32%, Ea error: 0.25%** |

### Tier 4 — Fe₂O₃ precursor problems (Problems 9–10)
One step from the real project.

| # | Problem | Physics encoded | Key skill | Result |
|---|---------|----------------|-----------|--------|
| 9 | Multi-cycle redox kinetics | Alternating reduction/oxidation over 3 cycles | Switching ODE with `tf.where`, boundary clustering | MAE: 0.0165, all continuity checks passed |
| 10 | Physics-data hybrid PINN | Shrinking core ODE + sparse TGA-style observations | Hybrid loss: `loss_physics + λ·loss_data`, three-model comparison | Model A (physics): Rel L2 = 0.0003 · Model B (data only): 0.34 · Model C (hybrid): 0.016 |

## Key technical lessons documented across the series

**DeepXDE (TensorFlow):**
- `dde.data.TimePDE` vs `dde.data.PDE` — critical distinction for time-dependent problems
- Temperature scaling (T/100) to prevent Arrhenius gradient explosion when E/RT > 10
- Never scale residuals inside the PDE function — use `loss_weights` at compile time
- Checking loss component magnitudes at epoch 0 before any tuning
- `dde.Variable` with log-scaling for multi-parameter inverse identification
- Per-temperature `t_max` selection for informative training data in inverse problems

**PyTorch (raw implementation):**
- `torch.clamp` on physical T range [100, 1000] K for Arrhenius — NOT on the exponent
- `t.grad.zero_()` inside training loop to prevent gradient accumulation across epochs
- IC weight must be proportional to PDE loss magnitude at epoch 0
- Coupled ODE system: two-output network, loss weight balancing across equations

**Problem 10 as preprint template:**

Problem 10 is the structural template for the Fe₂O₃ ChemRxiv preprint:

| Problem 10 | Fe₂O₃ PINN paper |
|-----------|-----------------|
| True shrinking core system | Fe₂O₃ reduction ODE |
| 8 sparse noisy observations | TGA experimental data |
| Model A (physics only) | Forward PINN baseline |
| Model B (data only) | ML regression baseline |
| Model C (hybrid) | ChemRxiv contribution |
| Error comparison table | Table 1 in paper |
| Discussion section | Section 4 in paper |

## Research context

This work is preparation for a Physics-Informed Neural Network for
Fe₂O₃ reduction kinetics in chemical looping gasification for hydrogen
production. The Fe₂O₃ redox PINN uses the shrinking core model trained
against published TGA experimental data, targeting a ChemRxiv preprint
submission in August 2026.

**Skills built in this series that transfer directly to the Fe₂O₃ project:**
ODE/PDE as physics residuals · Arrhenius kinetics encoding ·
Coupled nonlinear ODE systems · Multi-input parametric networks ·
Inverse parameter identification from noisy data ·
Physics-data hybrid loss formulation · Loss weight balancing ·
Numerical overflow protection for stiff Arrhenius systems

## Implementation

- **Framework:** DeepXDE (TensorFlow backend) + raw PyTorch
- **Problems 1–2:** DeepXDE only
- **Problem 3:** DeepXDE + raw PyTorch (both implementations included)
- **Problems 4–10:** DeepXDE

## Related repository

[1D-Heat-Equation-PINN](https://github.com/Kiran-1318/1D-Heat-Equation-PINN)
— Physics-Informed Neural Network for the 1D transient heat equation,
validated against analytical Fourier series solution. Relative L2 error: 0.99%.

## Author

**Kiran Thammina**
M.Tech Energy Systems Engineering, IIT Bombay (CPI 9.84, Best Thesis Award)
Project Engineer — Design, Organic Recycling Systems Limited
GitHub: [github.com/Kiran-1318](https://github.com/Kiran-1318)
