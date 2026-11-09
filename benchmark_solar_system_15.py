"""Benchmark copy of solar_system_15's workload, without printing full states.

Defaults to one simulated day at both 1-day and 1-second output increments.
These are dense-output intervals, not the adaptive integrator's step sizes.
"""

import argparse
import gc
import json
import math
import platform
import statistics
import time
from pathlib import Path

import plansim
import polars as pl

# Share the original example's exact physical constants and initial conditions.
from solar_system_15 import BODIES, DAY, initial_conditions


def benchmark(days=1.0, repeats=5, warmups=1):
    mus, rs, angles, vs, velocity_angles, _ = initial_conditions()
    results = []
    for spacing in (DAY, 1.0):
        solver = plansim.OrbitalSolver15(mus)
        timings = []
        for iteration in range(warmups + repeats):
            gc.collect()
            start = time.perf_counter()
            frame = solver.solve(rs, angles, vs, velocity_angles, (0.0, days * DAY), spacing)
            elapsed = time.perf_counter() - start
            assert isinstance(frame, pl.DataFrame)
            assert frame.columns == ["t"] + [str(i) for i in range(len(BODIES))]
            assert frame["t"][0] == 0.0
            assert math.isclose(frame["t"][-1], days * DAY, abs_tol=1e-6)
            assert frame.height >= 2
            for body in range(len(BODIES)):
                for field in ("x", "y", "z", "vx", "vy", "vz"):
                    assert frame[str(body)].struct.field(field).is_finite().all()
            rows = frame.height
            size = frame.estimated_size()
            del frame  # Never retain multiple large outputs between trials.
            if iteration >= warmups:
                timings.append(elapsed)
        result = dict(tspace_seconds=spacing, days=days, rows=rows,
                      output_bytes=size, seconds=timings,
                      median_seconds=statistics.median(timings),
                      min_seconds=min(timings))
        results.append(result)
        print(f"spacing={spacing:g}s rows={rows:,} median={result['median_seconds']:.6f}s "
              f"min={result['min_seconds']:.6f}s", flush=True)
    return dict(python=platform.python_version(), platform=platform.platform(),
                extension=plansim.__file__, results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.days) or args.days < 1:
        parser.error("--days must be finite and at least 1")
    if args.repeats < 1 or args.warmups < 0:
        parser.error("--repeats must be positive and --warmups nonnegative")
    report = benchmark(args.days, args.repeats, args.warmups)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
