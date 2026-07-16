# Plan Review: Full Multigroup Temperature Derivative

**Plan:** [`full_multigroup_derivative_d5139acd.plan.md`](/home/itamarg/.cursor/plans/full_multigroup_derivative_d5139acd.plan.md)  
**Reviewed revision:** the current deterministic + Monte Carlo revision  
**Verdict:** **Request changes.** The quotient-rule mathematics and most of the revised structure are sound, but two correctness/API issues must be resolved before implementation. The Monte Carlo strategy and numerical tests also need tightening to avoid a noisy or misleading validation.

## Findings

### P0 — The cutoff override cannot disable the configured cutoff

The plan gives `compute_matrix_impl` this parameter:

```cpp
std::optional<double> cutoff_override = std::nullopt
```

and defines `nullopt` as “use the stored `group_cutoff_ratio_`.” It then tries to disable cutoff in the full-derivative path by passing another `nullopt`:

```cpp
std::optional<double> const no_cutoff = std::nullopt;
compute_matrix_impl(..., no_cutoff);
```

Those states are indistinguishable. With the proposed selection logic, `no_cutoff` falls back to `group_cutoff_ratio_`, so a normally configured kernel still applies the heuristic cutoff. This contradicts the plan’s central correctness decision that all `G²` pairs are evaluated. See [plan lines 198–225](/home/itamarg/.cursor/plans/full_multigroup_derivative_d5139acd.plan.md:198) and [plan lines 251–276](/home/itamarg/.cursor/plans/full_multigroup_derivative_d5139acd.plan.md:251).

The proposed finite-difference tests instantiate `MGIntegrationConfig(cutoff_ratio=None)`, so they would accidentally hide this bug.

Use one of these unambiguous designs:

- Make the private method take the **effective cutoff** as a required `std::optional<double>`. Existing wrappers pass `group_cutoff_ratio_`; the full derivative passes `std::nullopt`.
- Add a separate `bool disable_cutoff` or a small enum.
- Use `std::optional<std::optional<double>>` as a tri-state override, although this is less readable.

Add a regression comparing `compute_dsigma_dT_matrix` from kernels configured with `cutoff_ratio=None` and a non-null cutoff. If the full path always disables cutoff, their results should agree to numerical precision.

### P0 — “Full derivative” is incomplete for supported temperature-dependent multipliers

Every proposed full-derivative API still accepts an arbitrary `KernelMultiplier`, and the plan says the weight-derivative multiplier composes with any user multiplier. However, for

\[
N(T)=\int w(E,T)\,\Sigma_E(T)\,m(E,E',\xi,T,N_e)\,dE\,dE'\,d\xi,
\]

the numerator derivative has a third contribution omitted by the plan:

\[
\frac{dN}{dT}
=\int \frac{\partial w}{\partial T}\Sigma_E m
+\int w\frac{\partial\Sigma_E}{\partial T}m
+\int w\Sigma_E\frac{\partial m}{\partial T}.
\]

This concern is now resolved: the `KernelMultiplier` interface no longer accepts
a temperature parameter (`operator()(E, Ep, xi, Ne)`), so temperature-dependent
multipliers cannot be expressed.  The derivative APIs are correct by
construction under this contract.

### P1 — The proposed MC three-stream estimator is unnecessarily noisy and is not unbiased “in expectation”

`mc_integrate` returns a self-normalized ratio using sampled `beta_avg` and `weight_avg`; see [compton_multigroup_monte_carlo.cpp](/home/itamarg/workspace_current/compton_cross_section/src/compton_multigroup/compton_multigroup_monte_carlo/compton_multigroup_monte_carlo.cpp:254). Such a finite-sample ratio estimator is generally biased, though consistent. Therefore the plan’s statement that three independently normalized calls are “correct in expectation” is too strong.

More importantly, the full derivative contains substantial cancellation. Estimating its three terms from independent streams makes the cancellation variance much worse, consumes three times the samples, and makes the proposed 5–10% MC-vs-deterministic CI assertion prone to flakiness—especially in the cold regime where the score contains `1/tau²`.

Prefer a single-pass MC score. For a temperature-independent multiplier, each contribution can use

\[
s_g = s_{\mathrm{kernel}} + \frac{\partial\ln w}{\partial T}
      - \frac{1}{D_g}\frac{dD_g}{dT}.
\]

The group-dependent denominator score can be precomputed and made available to the tally by passing `g0` to the internal callable. If the three-call design is retained, describe it as **consistent as sample count tends to infinity**, mark the high-sample validation as slow, and validate row sums or selected well-populated entries rather than every significant cell at a fixed relative tolerance.

### P1 — The proposed Planck expression is not stable for all positive `x`

The plan claims

```cpp
x * std::exp(x) / std::expm1(x)
```

is stable for all `x > 0`. It handles cancellation near zero, but for large finite `x`, both `exp(x)` and `expm1(x)` overflow and the ratio becomes `inf/inf -> NaN`. The class only validates `cap_x > 0`, so a cap above the floating-point exponential range is permitted.

Use the algebraically equivalent stable form

```cpp
x / (-std::expm1(-x))
```

which is well behaved at both small and large positive `x`.

Likewise, the claim that `weight(E,T) == 0` “never occurs for positive group boundaries” is false numerically: Planck/Wien weights can underflow when a large cap is configured. Computing `d_weight_dT / weight` also loses the benefit of an analytic logarithmic derivative. Prefer a `d_log_weight_dT` implementation for the internal multiplier, or explicitly constrain the supported cap range and test it. See [plan lines 93–104](/home/itamarg/.cursor/plans/full_multigroup_derivative_d5139acd.plan.md:93) and [plan lines 125–147](/home/itamarg/.cursor/plans/full_multigroup_derivative_d5139acd.plan.md:125).

### P1 — The validation plan needs cases that expose the new failure modes

The proposed tests are directionally good, especially cutoff-free deterministic FD and the independent integral check for `dD/dT`. Add or adjust these cases:

- **Cutoff override:** run the full method on a kernel with a non-null configured cutoff; otherwise the P0 sentinel bug passes unnoticed.
- **Multiplier contract:** multipliers are now temperature-independent by interface design; no FD check for `dm/dT` is needed.
- **Zero derivatives:** Planck/Wien entirely above the cap and Uniform have exact zero derivatives. A `1e-6` relative assertion is undefined at zero; use an absolute tolerance scaled to approximately `eps * D / h` for the FD residual, plus an exact/absolute check of the analytic result.
- **Cap split:** when checking `dD/dT = integral(d_weight_dT)`, give SciPy the moving cap energy as a split point (or integrate the two sides separately), because the pointwise derivative jumps there.
- **MC path:** MC-vs-deterministic uses the same analytic weight and denominator derivatives, so it is not a complete independent check. Add at least a same-seed Uniform test showing MC full derivative reduces to MC kernel-only derivative, or a common-random-number finite-difference check on a stable aggregate such as row sums.
- **MC runtime:** a 5–10 million-sample test becomes 15–30 million samples under the proposed three-call implementation. It should not be an unmarked default CI test.

### P2 — Public API and documentation details

- Adding pure virtual functions to `WeightFunction` is a source-breaking change for downstream C++ subclasses. That may be acceptable, but it should be called out. A non-pure default is not mathematically safe unless its contract is explicit.
- At exactly `E = cap_x * k_B * T`, the pointwise weight is continuous but generally not differentiable with respect to `T`. Document that `d_weight_dT` uses the capped-side convention (`0`) and that callers should not expect a two-sided derivative at the kink.
- The plan updates `IMPLEMENTATION.md` and headers, but the new public methods should also be added to the API summary in [README.md](/home/itamarg/workspace_current/compton_cross_section/README.md:180).
- If the MC three-call implementation remains, document that one public call advances the mutable RNG three times and therefore changes the subsequent RNG sequence relative to existing methods.

## What Is Correct in the Plan

- The quotient-rule decomposition is correct for the physical cross section, or for a multiplier held temperature-independent.
- The Planck and Wien pointwise derivative formulas are algebraically correct below the cap; Uniform derivatives are zero.
- The below-cap, above-cap, and straddling formulas for `d_denominator_dT` correctly account for the moving cap and continuity of `w`.
- Reusing the deterministic integrator through a `dln(w)/dT` multiplier is mathematically valid, subject to the underflow/stability note above.
- The revised deterministic design avoids copying the OpenMP traversal and aligns the C++/Python overload pattern with the existing API.
- Disabling the derivative cutoff is the right policy; derivative values can change sign or cancel, so sigma-style outward termination is not reliable.
- The magnitude-aware deterministic FD comparison and `integral(d_weight_dT)` denominator check are strong validation choices.

## Required Revisions

1. Fix the cutoff-control API so `nullopt` can genuinely mean “disabled,” and test a non-null configured cutoff.
2. Define whether `KernelMultiplier` is differentiated; either restrict/document the API or add the missing `dm/dT` term.
3. Replace the overflow-prone Planck ratio and address `d_weight/weight` underflow.
4. Rework the MC estimator toward a shared/single stream, or weaken and correctly characterize its statistical guarantees and CI test.
5. Add zero-derivative absolute tolerances, cap-split integration, MC-specific validation, and README documentation.

After those revisions, the plan is ready to implement; its core analytic approach is sound.
