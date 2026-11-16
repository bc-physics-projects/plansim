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


import argparse
import csv
import time
from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np
from astropy.time import Time
from astroquery.jplhorizons import Horizons


AU_KM = 149_597_870.7
DAY_S = 86_400.0
START_UTC = "2026-11-16 00:00:00"

# Each GM is for the *individual* body (not its system barycenter), in km^3/s^2.
# Source: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc
@dataclass(frozen=True)
class Body:
    name: str
    horizons_id: str
    gm_km3_s2: float


# Preserve the original script's exact 15-body order.
BODIES = (
    Body("Sun",      "10",  1.3271244004127942e11),
    Body("Jupiter",  "599", 1.266865319003704e8),
    Body("Saturn",   "699", 3.793120623436167e7),
    Body("Neptune",  "899", 6.835103145462294e6),
    Body("Uranus",   "799", 5.793951256527211e6),
    Body("Earth",    "399", 3.9860043550702266e5),
    Body("Venus",    "299", 3.2485859200000000e5),
    Body("Mars",     "499", 4.282837362069909e4),
    Body("Mercury",  "199", 2.2031868551400003e4),
    Body("Ganymede", "503", 9.887832752719638e3),
    Body("Titan",    "606", 8.978137095521046e3),
    Body("Callisto", "504", 7.179283402579837e3),
    Body("Io",       "501", 5.959915466180539e3),
    Body("Moon",     "301", 4.9028001184575496e3),
    Body("Europa",   "502", 3.202712099607295e3),
)
FIELDS = ("x", "y", "z", "vx", "vy", "vz")

def start_jd():
    """Convert a user-facing UTC timestamp to the TDB Julian Date Horizons expects."""
    return float(Time(START_UTC, format="iso", scale="utc").tdb.jd)


def jpl_states(julian_dates):
    """Retrieve barycentric states at TDB Julian Dates.

    Returns ndarray shape (number_of_epochs, 15, 6), with positions in AU
    and velocities in AU/day. Each body is queried independently, then kept
    in BODIES order. This works for moons without adding parent state vectors.
    """
    jds = [float(jd) for jd in julian_dates]
    if not jds:
        raise ValueError("At least one epoch is required")

    states = np.empty((len(jds), len(BODIES), 6), dtype=float)
    for i, body in enumerate(BODIES):
        print(f"Horizons: {body.name:<10} ({i + 1}/{len(BODIES)})", flush=True)
        # Keep each request short enough for Horizons' URL-based interface.
        for start in range(0, len(jds), 30):
            batch = jds[start:start + 30]
            table = Horizons(
                id=body.horizons_id,
                location="@0",            # Solar-system barycenter (SSB)
                epochs=batch,             # Julian Dates in TDB
            ).vectors(refplane="ecliptic", aberrations="geometric")

            if len(table) != len(batch):
                raise RuntimeError(
                    f"Horizons returned {len(table)} rows for {body.name}, expected {len(batch)}")
            returned_jds = np.asarray(table["datetime_jd"], dtype=float)
            if np.any(np.abs(returned_jds - batch) > 1e-6):
                raise RuntimeError(f"Horizons returned unexpected epochs for {body.name}")
            for j, field in enumerate(FIELDS):
                states[start:start + len(batch), i, j] = np.asarray(table[field], dtype=float)

    if not np.isfinite(states).all():
        raise ValueError("Horizons returned a missing or non-finite state value")
    return states

def initial_conditions():
    """Required output: (mus, (xs, ys, zs, vxs, vys, vzs)).

    Units: mus=>AU^3/day^2, positions=>AU, velocities=>AU/day.
    """
    mus = [body.gm_km3_s2 * DAY_S**2 / AU_KM**3 for body in BODIES]
    states = jpl_states([start_jd()])[0]  # shape: (15, 6)
    xs, ys, zs, vxs, vys, vzs = (states[:, j].tolist() for j in range(6))
    return mus, (xs, ys, zs, vxs, vys, vzs)

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

    solver = plansim.OrbitalSolver15(mus, kill_sigint=True)
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
