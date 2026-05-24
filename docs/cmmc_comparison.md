# CMMC Comparison and Known Artifacts

## Overview

This project validates the direct quadrature kernel against the CMMC (Compton Matrix Monte Carlo) code in `external/CMMC/`.  CMMC generates multigroup scattering matrices by Monte Carlo sampling of the Compton scattering process.

This document describes how the comparison works, the known artifacts of the CMMC redistribution scheme, and the regimes where agreement is expected.

---

## How CMMC Works

CMMC samples the scattering process:

1. For each incident energy group g₀, sample an incident photon energy E₀ from within the group.
2. Sample an electron from the Maxwell-Jüttner distribution at temperature T.
3. Compute the Compton scattering interaction (angle, outgoing energy E).
4. Accumulate the Thomson-weighted cross-section σ into the scattering matrix.

### The Redistribution Scheme

When the scattered photon lands in group g (found by binary search on energy boundaries), CMMC uses **linear interpolation** between the destination group g and the source group g₀:

```cpp
if (g == g0) {
    S[g0][g] += sigma;
} else {
    double fac = (E - E0) / (ec[g] - ec[g0]);
    S[g0][g]  += sigma * fac;
    S[g0][g0] += sigma * (1 - fac);
}
```

The intent: distribute σ between the destination and source groups proportionally to how close the actual energy transfer is to the nominal (center-to-center) energy transfer.

---

## The Diagonal Artifact

### Mechanism

When a scattered photon overshoots the destination bin center (E > ec[g] for g > g₀):

```
fac = (E - E₀) / (ec[g] - ec[g₀])  >  1
```

This makes `(1 - fac) < 0`, so the **diagonal element S[g₀][g₀] receives a negative contribution**.

For every scattering event that lands in a neighboring bin and overshoots its center, some cross-section is subtracted from the diagonal.

### When It Matters

The artifact is significant when the kernel broadening width is **larger than the bin width**:

| Condition | Broadening width | Effect |
|-----------|-----------------|--------|
| τ · E_in ≫ bin width | Many bins | Diagonal can become net **negative** |
| τ · E_in ~ bin width | A few bins | Diagonal is underestimated |
| τ · E_in ≪ bin width | Sub-bin | Artifact negligible |

Example: T = 100 keV, E_in = 300 keV:
- Doppler broadening ~ τ × E_in ~ 0.2 × 300 = 60 keV
- Bin width ~ 2.5 keV (for the standard 296-group grid)
- **Result:** MC diagonal = −2.7 mbarn/keV (should be +1.4 mbarn/keV)

### Properties of the Artifact

- **Systematic, not statistical:** Increasing MC samples converges to the biased value.
- **Conserves total cross-section:** The negative diagonal is compensated by excess in neighbors (total row sum is preserved).
- **Off-diagonal elements are correct:** Errors in individual off-diagonal elements average out.
- **Only the diagonal is severely affected:** It accumulates one-sided negative corrections from all neighbors.

### Impact on the Comparison Plots

On a log-scale plot, negative MC diagonal values simply disappear (matplotlib cannot plot negative values on log axes), creating an apparent "dip" or missing data point at E' ≈ E_in.

---

## Comparison Strategy

Given these limitations, the validation uses:

### 1. Wide energy groups (for MC comparison tests)

The tests in `tests/test_vs_mc.py` use wide bins (10–50 keV spacing) where:
- The redistribution artifact is mitigated (bin width ≈ broadening width)
- Statistical noise is reduced (more MC hits per bin)
- Both methods are well-resolved

### 2. Off-diagonal focus

For narrow-bin comparisons, focus on **off-diagonal elements** which are free of the redistribution artifact.  The diagonal should only be compared with wide bins or at low temperatures where broadening is sub-bin.

### 3. Low-temperature diagonal test

At T = 1 keV (τ ≈ 0.002), the Doppler broadening is much smaller than typical bins, so the diagonal redistribution artifact is negligible.  The `test_diagonal_1kev` test verifies diagonal agreement in this regime.

### 4. Tolerance levels

| Regime | Expected agreement | Tolerance used |
|--------|-------------------|----------------|
| T = 100 keV, wide bins | Good | 50% relative |
| T = 20 keV, wide bins | Good | 50% relative |
| T = 1 keV, diagonal only | Excellent | 5% relative |

---

## Angular Normalization

CMMC returns the **zeroth angular moment** of the scattering kernel:

```
S_CMMC[g, g'] = 2π · ∫₋₁¹ ∫_{E_lo}^{E_hi} Σ_E(E_g, E', ξ, τ) dE' dξ
```

This means the quadrature integration must use C_Ω = 2π (not 1/2 for the angular average).

---

## Regimes of Agreement

| Temperature | Off-diagonal | Diagonal | Notes |
|-------------|-------------|----------|-------|
| T = 100 keV | Excellent (< 10%) | **Poor** (artifact) | Wide bins fix diagonal |
| T = 20 keV | Good (< 20%) | Moderate | Less severe artifact |
| T = 1 keV | Exponentially small | Good (< 5%) | Broadening ≪ bin width |

---

## Generating Comparison Plots

```bash
python3 tests/plot_comparison.py
```

Outputs are saved to `tests/output/`:
- `quadrature_vs_mc_100kev.{png,pdf}`
- `quadrature_vs_mc_20kev_low.{png,pdf}`
- `quadrature_vs_mc_1kev_low.{png,pdf}`

The plots show σ(E') [mbarn/keV] vs E' for multiple incident energies, with:
- **Solid lines:** CMMC Monte Carlo (using `stairs` for bin-averaged data)
- **Dashed lines:** Direct quadrature (smooth, evaluated at bin centers)

---

## Potential CMMC Improvements

The redistribution artifact could be mitigated by:

1. **Clamping:** `fac = clamp(fac, 0, 1)` — sacrifices cross-section conservation but eliminates negativity.

2. **Sub-cell redistribution:** Only distribute within the destination bin and its immediate neighbor, using the bin edges (not centers) as interpolation anchors.

3. **Higher-order redistribution:** Use piecewise-linear or spline basis functions across multiple bins.

4. **Histogram deposit:** Simply add full σ to the destination bin g with no redistribution to g₀ — gives correct off-diagonal but loses the in-bin resolution.

These modifications would be changes to `external/CMMC/src/compton_matrix_mc.cpp`.
