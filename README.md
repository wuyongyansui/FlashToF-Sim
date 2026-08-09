# Flash dToF first-photon simulator

This is an independent, minimal Python project for the first executable link of
a Flash direct time-of-flight simulation platform. `SPCSimLib` was used only as
a read-only reference; this project does not import, copy, or modify its code.

The fixed version-0.1 chain is:

```text
user basic configuration
  -> ideal transient lambda[H,W,T]
  -> one earliest detected photon (or none) per pixel and laser period
  -> equi-width histogram counts[H,W,T]
  -> maximum-bin range d = c t / 2
  -> diagnostics
```

## Run

The prototype uses only the Python 3.8+ standard library.

The validated local environment is `SPCSimLib` with Python 3.8.20. It was
created with `conda create -n SPCSimLib python=3.8`; no third-party runtime
package is required by this project.

1. Edit the clearly marked `USER_CONFIG` block in `example_config.py`.
2. Optionally activate the validated environment with `conda activate
   SPCSimLib`.
3. From this directory, run:

```powershell
python run_simulation.py
```

Run the verification suite with:

```powershell
python -m unittest discover -s tests -v
```

No installation is required for these commands. An editable installation is
also possible with `python -m pip install -e .` when setuptools is available.

## Configuration contract

`UserConfig` contains only basic values the user is expected to edit. They are
validated immediately. `DerivedConfig` is read-only and contains consequences
such as round-trip time, pulse sigma, timing-window range, effective signal
flux, and tensor shapes.

The signal input is **expected detected photons per laser pulse** at unit
reflectivity and `reference_distance_m`. The first MVP uses the explicit toy
radiometric law

```text
effective signal = reference signal * reflectivity * (reference distance / distance)^2
```

so it does not hide an unconfigured photon-detection efficiency. Background is
expected detected photons per bin per pulse. All timing values are seconds and
all distances are metres.

The scene is intentionally homogeneous in version 0.1: every pixel has the same
distance, reflectivity, and flux. Explicit `H` and `W` axes make a later spatial
scene extension possible without changing the data contract.

## Shapes and units

| Quantity | Shape | Unit / meaning |
|---|---:|---|
| Ideal transient | `[H, W, T]` | expected detected photons / pixel / pulse / bin |
| First-photon outcomes | `[H, W, P]` | earliest bin index, or `-1` for no detection |
| EWH | `[H, W, T]` | first-photon event counts accumulated over `P` pulses |
| Peak bin and range | `[H, W]` | zero-based bin index and metres |

`T` is the number of equal-width timing bins and `P` is the number of laser
periods. An event tensor is retained in this small reference implementation for
auditability. A later large-scale implementation can stream the same outcomes
into the EWH.

## Why this is a true first-photon model

For independent Poisson arrivals with expected photons `lambda[k]` in bin `k`,
the probability that the first detected photon occurs in that bin is

```text
P(K=k) = exp(-sum(lambda[j], j<k)) * (1 - exp(-lambda[k]))
```

and the no-detection probability is `exp(-sum(lambda))`. The sampler draws once
from this categorical distribution for each pixel and laser period. It cannot
record a later bin after an earlier photon and cannot put one period into
multiple histogram bins. At low flux this distribution approaches the ideal
transient shape. At high flux its survival factor suppresses later arrivals,
producing the expected pile-up and earlier histogram peak.

This per-period acquisition gate is sufficient to generate pile-up. Cross-pulse
SPAD dead time is a separate effect and is not silently included.

## Modules

- `flash_dtof/config.py`: basic configuration, validation, derived quantities.
- `flash_dtof/transient.py`: integrated Gaussian signal plus uniform background.
- `flash_dtof/first_photon.py`: analytic first-arrival probabilities and seeded
  Monte Carlo sampling.
- `flash_dtof/ewh.py`: equi-width histogram accumulation and invariants.
- `flash_dtof/reconstruction.py`: maximum-bin ranging only.
- `flash_dtof/pipeline.py`: fixed chain orchestration and diagnostics.
- `tests/`: probability, reproducibility, shape/unit, and flux-trend checks.

## Deliberately not implemented

Version 0.1 does **not** include pile-up inversion or correction (Coates or
otherwise), leading-edge, centroid, matched-filter, or sub-bin reconstruction,
TDC offset/DNL/jitter, cross-period dead time, complex spatial scenes, multipath,
or multiple returns. These remain explicit later extension points rather than
unverified effects hidden in the baseline.
