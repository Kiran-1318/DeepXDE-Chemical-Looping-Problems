# DeepXDE Chemical Looping Problems

A series of 10 original Physics-Informed Neural Network (PINN) problems
designed as a deliberate progression toward a PINN for Fe₂O₃ redox
kinetics in chemical looping gasification for hydrogen production.

All problems are implemented from scratch — not tutorial replications.
Each problem teaches one specific skill needed for the core Fe₂O₃ project.

## Problem series

### Tier 1 — ODE kinetics (Problems 1–3)
Builds the mathematical skeleton for solid-gas kinetic modelling.

| # | Problem | Key skill |
|---|---------|-----------|
| 1 | Isothermal first-order batch reactor | ODE as physics residual, Adam  |
| 2 | Arrhenius temperature-dependent rate | Encoding k(T) = A·exp(−Ea/RT) in loss |
| 3 | Non-isothermal coupled reactor | Coupled ODE system, loss weight balancing |

### Tier 2 — Solid-gas conversion (Problems 4–6)
*Coming soon*

### Tier 3 — Inverse problems (Problems 7–8)
*Coming soon*

### Tier 4 — Fe₂O₃ precursor problems (Problems 9–10)
*Coming soon*

## Key technical lessons documented

- `dde.data.TimePDE` vs `dde.data.PDE` — critical distinction for
  time-dependent problems
- Temperature scaling (T/100) to prevent Arrhenius gradient explosion
  when E/RT > 10
- Loss weight balancing — never scale residuals inside the PDE function,
  use `loss_weights` at compile time
- `torch.clamp` on physical T range [100, 1000] K, not on the exponent —
  clamping the exponent allows positive values when T < 0, causing
  loss explosion to 10¹⁴
- `t.grad.zero_()` inside PyTorch training loop — prevents gradient
  accumulation across epochs
- Checking loss component magnitudes at epoch 0 before any tuning

## Research context

This work is preparation for a Physics-Informed Neural Network for
Fe₂O₃ reduction kinetics using the shrinking core model, with validation
against TGA experimental data. The final project targets a ChemRxiv
preprint connecting to PhD research in chemical looping gasification.

## Implementation

- Framework: DeepXDE (TensorFlow backend) + raw PyTorch
- Problems 1–2: DeepXDE only
- Problem 3 onwards: both DeepXDE and PyTorch implementations

## Author

Kiran Thammina
M.Tech Energy Systems Engineering, IIT Bombay
GitHub: github.com/Kiran-1318
