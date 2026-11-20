# plansim

Build and install the Rust extension with `maturin develop --release` in your Python
environment. Installation also installs the required Python Polars package.

`OrbitalSolver15` and `OrbitalSolver90` through `OrbitalSolver100` expose
`solve(xs, ys, zs, vxs, vys, vzs, tspan, tspace)` and return a `polars.DataFrame`.
Each row is one time sample:

- `t`: sample time, Float64.
- `"0"`, `"1"`, ...: zero-based body numbers in input order. Each column is a
  Struct with Float64 fields `x`, `y`, `z`, `vx`, `vy`, `vz`.

Polars uses nested Struct columns rather than a pandas-style MultiIndex.

```python
df = solver.solve(xs, ys, zs, vxs, vys, vzs, (0.0, 100.0), 1.0)
times = df["t"]
x0 = df["0"].struct.field("x")
body0 = df["0"].struct.unnest()  # six-column DataFrame
```

This replaces the previous `OrbitalSolverResults*` classes and their `get_*`
methods. Run the integration checks after installation with
`python -m unittest discover -s tests`.

## Performance and profiling

Release builds optimize for speed (`opt-level = 3`, fat LTO, one codegen unit).
The integrator stores states as six contiguous component arrays (structure of
arrays): `[x[0..N], y[0..N], z[0..N], vx[0..N], vy[0..N], vz[0..N]]`.
The force kernel uses explicit `std::simd::f64x4` batches for every system size,
with scalar tails for leftover pairs. Each pair shares a distance calculation,
and contributions are summed in their original order without fast-math.
This uses the existing nightly toolchain and its `portable_simd` feature.
The derivative callback does not allocate on the heap.
Python inputs and per-body DataFrame fields retain their existing layout.
Gravity evaluates each unordered pair once. Integration releases the Python
GIL, and output uses NumPy-backed numeric buffers instead of Python float lists.
The original example remains in `solar_system_15.py`; its benchmark counterpart
shares the same initial conditions and excludes printing and validation from timing.

From an activated project virtual environment:

```powershell
maturin develop --release
python benchmark_solar_system_15.py --json target/benchmarks/release.json
rustup component add llvm-tools
maturin build --release --pgo -i python --out target/pgo-wheels
python -m pip install --force-reinstall --no-deps (Get-ChildItem target/pgo-wheels/*.whl).FullName
python benchmark_solar_system_15.py --json target/benchmarks/pgo.json
python -m unittest discover -s tests
```

Maturin 1.16+ runs the configured `pgo-command` in a temporary environment using
an instrumented wheel, merges the LLVM profiles, then rebuilds using those profiles.
See [Maturin's PGO configuration](https://www.maturin.rs/config#pgo-command).
Training covers the 15-body workload at both output intervals. Rerun PGO after
changing the Rust code or representative workload; it does not guarantee a gain
for the 90–100-body solvers. The wheel must be installed to benchmark PGO;
`maturin develop` would replace it with an ordinary release build.

The benchmark defaults to one simulated day, with output every 86,400 seconds
and every second (2 and 86,401 rows). `--days`, `--repeats`, and `--warmups`
control duration and repetitions. These are output intervals, not fixed integration
steps. The solver still uses DOP853 with rtol=1e-8 and atol=1e-6. Timings include
integration and DataFrame construction, with input setup outside the timed region.
Large durations at one-second spacing consume substantial memory (at least
728 bytes per output row, plus integrator storage).

For native profiling with symbols, use `maturin develop --profile profiling`.
For a wheel specific to the build machine, optionally set
`$env:RUSTFLAGS = "-C target-cpu=native"` before building and benchmarking.
The default build keeps portable CPU targeting.

### Local benchmark results

Measured on Windows x86-64, CPython 3.12.14, Rust 1.101 nightly, Polars 2.1.0,
and NumPy 2.5.4. One simulated day; median wall time including DataFrame creation:

| Output interval | Original size-optimized release | Optimized release | Optimized + PGO |
| --- | ---: | ---: | ---: |
| 1 day (2 rows) | 3.320 ms | 2.104 ms | 2.033 ms |
| 1 second (86,401 rows) | 456.319 ms | 253.407 ms | 237.911 ms |

Original and release measurements used 5 timed runs after 1 warmup; the final
PGO confirmation used 15 runs after 3 warmups because the initial short run was
noisy. These are local observations, not guaranteed speedups. In particular,
millisecond-scale measurements and the small incremental PGO difference are
sensitive to system load. Raw timing reports are generated under
`target/benchmarks/{baseline,optimized,pgo}.json` (ignored build artifacts).
The final PGO wheel is installed in the project `.venv`.

Validation passed: Rust direct-force comparison and momentum conservation;
Python schema/constant-velocity tests for 15, 90, and 100 bodies; daily/second
sampling agreement; finite outputs; invalid time inputs; and comparison with
the original solar trajectory at rtol=1e-8, atol=1e-6. `cargo clippy --all-targets
-- -D warnings` and `cargo fmt --check` also passed.
