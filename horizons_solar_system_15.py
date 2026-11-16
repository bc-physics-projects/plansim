# %% Imports

import argparse
import time
from math import isfinite

import matplotlib.pyplot as plt
import polars as pl

import plansim

if not hasattr(plansim, "OrbitalSolver15"):
    raise SystemExit(
        "Rebuild/install the local plansim extension with OrbitalSolver15 enabled."
    )
# %% Initial condition definitions

BODIES = [
    "Sun",
    "Jupiter",
    "Saturn",
    "Neptune",
    "Uranus",
    "Earth",
    "Venus",
    "Mars",
    "Mercury",
    "Ganymede",
    "Titan",
    "Callisto",
    "Io",
    "Moon",
    "Europa",
]


def initial_conditions():
    """
    Required output: (mus, (xs, ys, zs, vxs, vys, vzs))
    Units: mus=>AU^3/day^2, positions=>AU, velocities=>AU/day
    """
    raise NotImplementedError


# %% Actual solver and plotting code.


def plot_points(result):
    """Display every sampled position of the Sun, planets, and moons in x/y."""

    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    colors = plt.get_cmap("tab20").colors
    for i, body in enumerate(BODIES):
        state = result[str(i)]
        x = state.struct.field("x").to_numpy()
        y = state.struct.field("y").to_numpy()
        ax.plot(
            x,
            y,
            linestyle="solid",
            marker=None,
            markersize=2,
            color=colors[i],
            label=body.name,
            rasterized=True,
        )

    ax.set(
        xlabel="Barycentric x [AU]",
        ylabel="Barycentric y [AU]",
        title=f"Sun, planet, and moon positions over {result['t'][-1]:g} days",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), markerscale=4)
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=30.0)
    parser.add_argument(
        "--tspace",
        type=float,
        default=1.0,
        help="Time between samples in days (default: 1)",
    )
    parser.add_argument(
        "--initial-only", action="store_true", help="Print initial data without solving"
    )
    args = parser.parse_args()
    if not isfinite(args.days) or args.days <= 0:
        parser.error("--days must be finite and positive")
    if not isfinite(args.tspace) or args.tspace <= 0:
        parser.error("--tspace must be finite and positive")

    mus, init_data = initial_conditions()

    solver = plansim.OrbitalSolver15(mus)
    start = time.perf_counter()
    result = solver.solve(*init_data, (0.0, args.days), args.tspace)
    elapsed = time.perf_counter() - start
    if not isinstance(result, pl.DataFrame):
        raise SystemExit("Rebuild/install plansim for the Polars DataFrame API.")
    times = result["t"]
    print(f"Solver call time: {elapsed:.3f} seconds.")
    print(f"Final time: {times[-1]:g} days.")
    plot_points(result)
    return result


if __name__ == "__main__":
    main()
