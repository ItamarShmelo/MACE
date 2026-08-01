# Examples

Install MACE from the repository root before running these examples:

```bash
uv sync
uv pip install -e .
```

Then run any script from the repository root:

```bash
uv run python examples/point_kernel.py
uv run python examples/multigroup_matrix.py
uv run python examples/monte_carlo_comparison.py
```

The scripts write plots and NumPy data files to `examples/output/`.
The deterministic multigroup calculations can take longer than the pointwise
example because they evaluate nested energy and angular quadratures.
