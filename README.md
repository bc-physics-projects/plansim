# plansim

Build and install the Rust extension with `maturin develop` in your Python
environment. Installation also installs the required Python Polars package.

`OrbitalSolver15` and `OrbitalSolver90` through `OrbitalSolver100` expose
`solve(rs, r_angs, vs, v_angs, tspan, tspace)` and return a `polars.DataFrame`.
Each row is one time sample:

- `t`: sample time, Float64.
- `"0"`, `"1"`, ...: zero-based body numbers in input order. Each column is a
  Struct with Float64 fields `x`, `y`, `z`, `vx`, `vy`, `vz`.

Polars uses nested Struct columns rather than a pandas-style MultiIndex.

```python
df = solver.solve(rs, r_angs, vs, v_angs, (0.0, 100.0), 1.0)
times = df["t"]
x0 = df["0"].struct.field("x")
body0 = df["0"].struct.unnest()  # six-column DataFrame
```

This replaces the previous `OrbitalSolverResults*` classes and their `get_*`
methods. Run the integration checks after installation with
`python -m unittest discover -s tests`.
