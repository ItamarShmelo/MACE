"""
Shared utilities for the mega-validation report suite.

Provides grid definitions, weight-function constructors, MC multi-seed
runner, mixed abs+rel error metrics, checkpoint/resume helpers, a
wall-clock timing accumulator, progress printing, and plot-tier logic.
"""
import argparse
import os
import sys
import time
from contextlib import contextmanager

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# ── path setup ────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'cpp_modules'))

import _compton_multigroup as cm                                    # noqa: E402
from _compton_differential_cross_section import ComptonKernelSolver  # noqa: E402
from _units import kev, kev_kelvin, sigma_thomson                   # noqa: E402

# ── physical constants re-exported for convenience ────────────────────────

KEV = kev
KEV_KELVIN = kev_kelvin
SIGMA_T = sigma_thomson

KERNEL = ComptonKernelSolver()

# ── temperature grid (25 points, 1e-5 keV … 1e4 keV) ─────────────────────

TEMPS_KEV = np.geomspace(1e-5, 1e4, 25)

# ── energy-group grids (6 types, all fixed boundary arrays in keV) ────────

def _nonuniform_grid():
    """G24-nonuniform: denser in [0.01, 1000] keV."""
    seg1 = np.logspace(-5, -2, 7)       # 6 groups
    seg2 = np.logspace(-2, 3, 13)       # 12 groups
    seg3 = np.logspace(3, 4, 7)         # 6 groups
    return np.unique(np.concatenate([seg1, seg2, seg3]))


def _hybrid_grid():
    """G28-hybrid: log / linear / log."""
    seg1 = np.logspace(-5, 0, 11)       # 10 groups
    seg2 = np.linspace(1.0, 100.0, 11)  # 10 groups
    seg3 = np.logspace(2, 4, 9)         # 8 groups
    return np.unique(np.concatenate([seg1, seg2, seg3]))


GRIDS = [
    {'name': '16-group log [1e-5, 1e4] keV',
     'tag': 'G16log', 'bounds_kev': np.logspace(-5, 4, 17)},
    {'name': '32-group log [1e-5, 1e4] keV',
     'tag': 'G32log', 'bounds_kev': np.logspace(-5, 4, 33)},
    {'name': '48-group log [1e-5, 1e4] keV',
     'tag': 'G48log', 'bounds_kev': np.logspace(-5, 4, 49)},
    {'name': '24-group non-uniform',
     'tag': 'G24nu', 'bounds_kev': _nonuniform_grid()},
    {'name': '28-group hybrid (log/lin/log)',
     'tag': 'G28hyb', 'bounds_kev': _hybrid_grid()},
    {'name': '20-group log [1e-5, 1e4] keV',
     'tag': 'G20log', 'bounds_kev': np.logspace(-5, 4, 21)},
]

# ── weight functions ──────────────────────────────────────────────────────

WEIGHT_SPECS = [
    ('planck', lambda: cm.PlanckWeightFunction(cap_x=200.0)),
    ('wien',   lambda: cm.WienWeightFunction(cap_x=25.0)),
    ('uniform', lambda: cm.UniformWeightFunction()),
]

def make_weight(name):
    """Return a fresh weight-function object by name."""
    for wn, factory in WEIGHT_SPECS:
        if wn == name:
            return factory()
    raise ValueError(f'Unknown weight function: {name}')

# ── MC configuration ─────────────────────────────────────────────────────

MC_SEEDS = [42, 137, 271, 577, 1009, 1543, 2027, 2741, 3571, 4219]
MC_SAMPLES = 5_000_000

# ── representative subsets for plotting ───────────────────────────────────

REPR_GRID_TAGS = {'G32log', 'G24nu', 'G28hyb'}
REPR_TEMPS_KEV = np.geomspace(1e-5, 1e4, 10)

def is_representative(grid_tag, T_kev, plot_tier):
    """Return True if this case should get per-case plots."""
    if plot_tier == 'full':
        return True
    if plot_tier == 'smoke':
        return False
    if grid_tag not in REPR_GRID_TAGS:
        return False
    return any(np.isclose(T_kev, rt, rtol=0.01) for rt in REPR_TEMPS_KEV)

# ── factory helpers ───────────────────────────────────────────────────────

def make_det(bounds_kev, wf, config=None):
    if config is None:
        config = cm.MGIntegrationConfig(
            base_order=24,
            peak_max_depth=5,
            integration_tolerance=1e-3,
            cutoff_ratio=1e-8,
        )
    return cm.ComptonMultigroupKernel(
        energy_group_boundaries=(bounds_kev * KEV).tolist(),
        weight_function=wf,
        config=config,
    )


def make_mc(bounds_kev, wf, num_samples, seed):
    return cm.ComptonMonteCarloKernel(
        energy_group_boundaries=(bounds_kev * KEV).tolist(),
        weight_function=wf,
        config=cm.MCIntegrationConfig(
            num_samples=num_samples,
            seed=seed,
            discard_out_of_grid=True,
        ),
    )

# ── MC multi-seed ensemble runner ─────────────────────────────────────────

def run_mc_ensemble(bounds_kev, wf, T_K, num_samples, seeds, compute_func):
    """
    Run *compute_func* for each seed and return statistics.

    Parameters
    ----------
    compute_func : str
        ``'sigma'`` or ``'dsigma'``.

    Returns
    -------
    mean, std, two_sigma, stack : np.ndarray
    """
    results = []
    for seed in seeds:
        mc_obj = make_mc(bounds_kev, wf, num_samples, seed)
        if compute_func == 'sigma':
            S = np.array(mc_obj.compute_sigma_matrix(T=T_K, Ne=1.0))
        elif compute_func == 'dsigma':
            S = np.array(mc_obj.compute_dsigma_dT_matrix(T=T_K, Ne=1.0))
        else:
            raise ValueError(compute_func)
        G = len(bounds_kev) - 1
        results.append(S.reshape(G, G))
    stack = np.stack(results)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(mean)
    two_sigma = 2.0 * std
    return mean, std, two_sigma, stack

# ── error metrics ─────────────────────────────────────────────────────────

def mixed_error(test, ref, abs_floor):
    """
    Element-wise mixed abs+rel error.

    Returns ``(error_array, masked_fraction)`` where *masked_fraction* is
    the share of elements whose ``|ref|`` falls below *abs_floor*.
    """
    denom = np.maximum(np.abs(ref), abs_floor)
    err = np.abs(test - ref) / denom
    masked_frac = float(np.mean(np.abs(ref) < abs_floor))
    return err, masked_frac


def error_stats(err):
    """Return (max, mean, median, p95) of finite elements in *err*."""
    finite = err[np.isfinite(err)]
    if len(finite) == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (float(np.max(finite)), float(np.mean(finite)),
            float(np.median(finite)), float(np.percentile(finite, 95)))


def row_sums(S):
    return S.sum(axis=-1)

# ── checkpoint / resume ───────────────────────────────────────────────────

def cache_key(cache_dir, grid_tag, T_kev, weight_name, suffix=''):
    return os.path.join(
        cache_dir,
        f'{grid_tag}_T{T_kev:.6g}_{weight_name}{suffix}.npz',
    )


def save_checkpoint(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_checkpoint(path):
    if os.path.exists(path):
        try:
            return dict(np.load(path, allow_pickle=False))
        except Exception:
            return None
    return None

# ── timing accumulator ────────────────────────────────────────────────────

class TimingLog:
    """Accumulate wall-clock durations by label."""

    def __init__(self):
        self._records: dict[str, float] = {}
        self._start: float | None = None
        self._label: str = ''

    @contextmanager
    def __call__(self, label):
        t0 = time.perf_counter()
        yield
        dt = time.perf_counter() - t0
        self._records[label] = self._records.get(label, 0.0) + dt

    def get(self, label):
        return self._records.get(label, 0.0)

    def items(self):
        return self._records.items()

    def total(self):
        return sum(self._records.values())

# ── progress printing ─────────────────────────────────────────────────────

_T0 = time.time()


def progress(part_num, msg):
    elapsed_h = (time.time() - _T0) / 3600.0
    print(f'[Part {part_num}] {msg} | {elapsed_h:.2f}h elapsed', flush=True)

# ── report writing helpers ────────────────────────────────────────────────

def emit(lines, s=''):
    lines.append(s)


def save_fig(figs_dir, gen_dir, name):
    path = os.path.join(figs_dir, name)
    plt.savefig(path, dpi=140, bbox_inches='tight')
    plt.close()
    return os.path.relpath(path, gen_dir)


def write_report(gen_dir, filename, lines):
    os.makedirs(gen_dir, exist_ok=True)
    path = os.path.join(gen_dir, filename)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Wrote {path}', flush=True)

# ── CLI helpers ───────────────────────────────────────────────────────────

def base_arg_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--plot-tier', choices=['smoke', 'standard', 'full'],
                   default='standard',
                   help='Plotting verbosity tier (default: standard)')
    p.add_argument('--no-cache', action='store_true',
                   help='Ignore existing checkpoints and recompute everything')
    return p

# ── failure-budget helper ─────────────────────────────────────────────────

class FailureTracker:
    """Track per-case failures and apply a 5 %% budget."""

    def __init__(self, total_cases):
        self.total = total_cases
        self.failures: list[dict] = []

    def record(self, grid_tag, T_kev, weight_name, exc):
        self.failures.append({
            'grid': grid_tag, 'T_kev': T_kev,
            'weight': weight_name, 'error': str(exc),
        })

    @property
    def count(self):
        return len(self.failures)

    @property
    def fraction(self):
        return self.count / max(self.total, 1)

    @property
    def exit_code(self):
        return 1 if self.fraction > 0.05 else 0

    def emit_section(self, lines):
        if not self.failures:
            emit(lines, '## Failures')
            emit(lines, '')
            emit(lines, 'None.')
            emit(lines)
            return
        emit(lines, '## Failures')
        emit(lines, '')
        emit(lines, f'{self.count}/{self.total} cases failed '
             f'({100*self.fraction:.1f}%).')
        emit(lines, '')
        emit(lines, '| Grid | T (keV) | Weight | Error |')
        emit(lines, '|------|---------|--------|-------|')
        for f in self.failures:
            emit(lines, f"| {f['grid']} | {f['T_kev']:.4g} "
                 f"| {f['weight']} | {f['error'][:80]} |")
        emit(lines)
