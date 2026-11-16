"""15-body solar-system simulation: JPL initial state -> Rust solver -> JPL check.

Body order: Sun, eight planets (same as the original script), and six moons.
Coordinates: JPL solar-system barycentric, J2000 ecliptic, geometric vectors.
Units everywhere in the solver: AU, days, AU/day, AU^3/day^2.

Dependencies:
    pip install numpy astroquery astropy polars matplotlib
    pip install ellipse           # only for --fit-ellipses or --fitplot
    # Also install/rebuild your local plansim Rust extension.

Examples:
    python solar_system_15_jpl.py --initial-only
    python solar_system_15_jpl.py --days 365 --tspace 0.02
    python solar_system_15_jpl.py --days 365 --tspace 0.02 --checks 13
    python solar_system_15_jpl.py --days 365 --no-plot
    python solar_system_15_jpl.py --days 365 --fitplot

Important: This expects the updated Rust solver signature:
    OrbitalSolver15(mus).solve(xs, ys, zs, vxs, vys, vzs,
                              (start_day, end_day), sample_step_days)
The code NEVER supplies future Horizons positions to the solver.
"""

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


def state_array(result, body_index):
    """Read a body's Cartesian states from the Rust solver's Polars DataFrame."""
    series = result[str(body_index)]
    return np.column_stack([
        series.struct.field(field).to_numpy() for field in FIELDS
    ]).astype(float)


def print_initial(mus, initial_state):
    print(f"\nJPL initial conditions at {START_UTC} UTC")
    print("Solar-system barycenter, ecliptic J2000; AU, AU/day, AU^3/day^2")
    print(f"{'Body':<10} {'mu':>14} {'x':>13} {'y':>13} {'z':>13} {'vx':>13} {'vy':>13} {'vz':>13}")
    states = np.asarray(initial_state, dtype=float).T
    for body, mu, state in zip(BODIES, mus, states):
        print(f"{body.name:<10} {mu:14.7e} " + " ".join(f"{v:13.6e}" for v in state))


def compare_to_jpl(result, checks, csv_path):
    """Evaluate deviations at solver sample times. Never feed these back to Rust."""
    sim_times = np.asarray(result["t"].to_numpy(), dtype=float)
    if len(sim_times) < 2 or np.any(np.diff(sim_times) <= 0):
        raise ValueError("The solver must return at least two increasing time samples")

    # Pick real output indices, avoiding interpolation or mismatched dates.
    indices = np.unique(np.rint(np.linspace(0, len(sim_times) - 1,
                                               min(checks + 1, len(sim_times)))).astype(int))
    days = sim_times[indices]
    print(f"\nChecking against Horizons at {len(indices)} matching simulation times ...")
    reference = jpl_states(start_jd() + days)

    position_error_km = np.empty((len(indices), len(BODIES)))
    velocity_error_kms = np.empty_like(position_error_km)
    for i, body in enumerate(BODIES):
        computed = state_array(result, i)[indices]
        delta = computed - reference[:, i, :]
        position_error_km[:, i] = np.linalg.norm(delta[:, :3], axis=1) * AU_KM
        velocity_error_kms[:, i] = np.linalg.norm(delta[:, 3:], axis=1) * AU_KM / DAY_S

    print(f"\nPosition error at t={days[-1]:.4f} days:")
    print(f"{'Body':<12} {'Error [km]':>17} {'Velocity error [km/s]':>24}")
    for i, body in enumerate(BODIES):
        print(f"{body.name:<12} {position_error_km[-1, i]:17.6f} {velocity_error_kms[-1, i]:24.9e}")

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_days", "body", "position_error_km", "velocity_error_km_s"])
        for j, day in enumerate(days):
            for i, body in enumerate(BODIES):
                writer.writerow([float(day), body.name,
                                 float(position_error_km[j, i]),
                                 float(velocity_error_kms[j, i])])
    print(f"\nComparison results saved to {csv_path}")
    return days, position_error_km


def plot_points(result):
    """Plot 15 simulated trajectories in the ecliptic XY projection (AU)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    colors = plt.get_cmap("tab20").colors
    for i, body in enumerate(BODIES):
        values = state_array(result, i)
        ax.plot(values[:, 0], values[:, 1], lw=0.9, color=colors[i], label=body.name)
    ax.set(xlabel="SSB x [AU]", ylabel="SSB y [AU]",
           title=f"Our 15-body simulation: {float(result['t'][-1]):g} days")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    return fig


def plot_errors(days, position_errors):
    """Plot the position discrepancy vs JPL (km) for each of the 15 bodies."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    colors = plt.get_cmap("tab20").colors
    for i, body in enumerate(BODIES):
        ax.plot(days, position_errors[:, i], label=body.name, color=colors[i])
    ax.set(xlabel="Time since start [days]", ylabel="3D position error [km]",
           title="Simulation error relative to JPL Horizons")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    return fig


def fit_ellipse_parameters(x, y):
    """Fit the apparent 2D XY trajectory to an ellipse (optional diagnostic)."""
    from ellipse import LsqEllipse

    points = np.column_stack((x, y))
    if len(points) < 5 or not np.isfinite(points).all():
        raise ValueError("At least five finite samples are needed")
    origin = points.mean(axis=0)
    centered = points - origin
    _, scales, rotation = np.linalg.svd(centered, full_matrices=False)
    if scales[-1] <= scales[0] * 1e-12:
        raise ValueError("Samples are coincident or nearly collinear")
    scales /= sqrt(len(points))
    transform = rotation.T @ np.diag(scales)
    fitted = LsqEllipse().fit((centered @ rotation.T) / scales)
    center, a, b, theta = fitted.as_parameters()
    parameters = np.real_if_close(np.array([*center, a, b, theta]))
    if np.iscomplexobj(parameters) or not np.isfinite(parameters).all():
        raise ValueError("No finite ellipse fit")
    center, a, b, theta = parameters[:2], *parameters[2:]
    if min(a, b) <= 0:
        raise ValueError("Ellipse axes must be positive")
    center = transform @ center + origin
    ct, st = np.cos(theta), np.sin(theta)
    axes = transform @ np.array([[ct, -st], [st, ct]]) @ np.diag([a, b])
    directions, lengths, _ = np.linalg.svd(axes)
    a, b = lengths
    theta = np.arctan2(directions[1, 0], directions[0, 0])
    return center, a, b, theta


def ellipse_points(center, a, b, theta, n=500):
    angle = np.linspace(0, 2 * np.pi, n)
    ct, st = np.cos(theta), np.sin(theta)
    return (center[0] + a * np.cos(angle) * ct - b * np.sin(angle) * st,
            center[1] + a * np.cos(angle) * st + b * np.sin(angle) * ct)


def plot_ellipses(result):
    """Fit ellipses to all eight planetary XY curves, as in original script."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    print("\nPlanet ellipse fit errors (normalized ellipse-equation residuals):")
    print(f"{'Planet':<10} {'MSE':>14} {'RMSE':>14}")
    colors = plt.get_cmap("tab20").colors
    for i, body in enumerate(BODIES[1:9], 1):
        values = state_array(result, i)
        x, y = values[:, 0], values[:, 1]
        try:
            (xc, yc), a, b, theta = fit_ellipse_parameters(x, y)
        except (ValueError, IndexError, np.linalg.LinAlgError) as exc:
            print(f"{body.name:<10} Fit unavailable: {exc}")
            continue
        dx, dy = x - xc, y - yc
        ct, st = np.cos(theta), np.sin(theta)
        residuals = ((dx * ct + dy * st) / a)**2 + ((-dx * st + dy * ct) / b)**2 - 1
        mse = float(np.mean(residuals**2))
        print(f"{body.name:<10} {mse:14.6e} {np.sqrt(mse):14.6e}")
        fx, fy = ellipse_points((xc, yc), a, b, theta)
        ax.plot(fx, fy, label=body.name, color=colors[i])
    ax.set(xlabel="SSB x [AU]", ylabel="SSB y [AU]", title="Fitted planetary XY ellipses")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    return fig


def fitplot(result):
    """Show simulation data alongside a fitted 2D ellipse for each planet."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(17, 9), layout="constrained")
    for i, (body, ax) in enumerate(zip(BODIES[1:9], axes.flat), start=1):
        values = state_array(result, i)
        x, y = values[:, 0], values[:, 1]
        ax.plot(x, y, ".", ms=2, label="Simulation")
        try:
            center, a, b, theta = fit_ellipse_parameters(x, y)
            ex, ey = ellipse_points(center, a, b, theta)
            ax.plot(ex, ey, "--", label="Ellipse fit")
        except (ValueError, IndexError, np.linalg.LinAlgError) as exc:
            ax.text(0.03, 0.97, f"Fit unavailable: {exc}", transform=ax.transAxes,
                    va="top", fontsize=8)
        ax.set(title=body.name, xlabel="x [AU]", ylabel="y [AU]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Projected XY orbit ellipse fits (short arcs may be unreliable)")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=365.0,
                        help="Simulation duration in days (default 365)")
    parser.add_argument("--tspace", type=float, default=0.02,
                        help="Solver sampling interval in days (default 0.02, about 29 minutes)")
    parser.add_argument("--checks", type=int, default=13,
                        help="Number of JPL comparisons after t=0 (default 13)")
    parser.add_argument("--initial-only", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--fit-ellipses", action="store_true")
    parser.add_argument("--fitplot", action="store_true")
    parser.add_argument("--csv", default="jpl_comparison.csv")
    args = parser.parse_args()
    if not isfinite(args.days) or args.days <= 0:
        parser.error("--days must be finite and > 0")
    if not isfinite(args.tspace) or args.tspace <= 0:
        parser.error("--tspace must be finite and > 0")
    if args.checks < 1:
        parser.error("--checks must be >= 1")

    mus, initial_state = initial_conditions()
    print_initial(mus, initial_state)
    if args.initial_only:
        return

    import polars as pl
    import plansim

    if not hasattr(plansim, "OrbitalSolver15"):
        raise SystemExit("Rebuild/install your local plansim extension with OrbitalSolver15")

    solver = plansim.OrbitalSolver15(mus)
    start = time.perf_counter()
    # Updated Cartesian solver; simulation time in DAYS, not seconds.
    result = solver.solve(*initial_state, (0.0, args.days), args.tspace)
    elapsed = time.perf_counter() - start

    if not isinstance(result, pl.DataFrame):
        raise TypeError("Expected a Polars DataFrame from the updated Rust solver")
    print(f"\nRust solver finished: {result.height} samples, "
          f"final time {float(result['t'][-1]):.6f} days, {elapsed:.3f} seconds elapsed")

    final_states = pl.concat([
        result[str(i)].tail(1).struct.unnest().with_columns(
            pl.lit(i).alias("mass #"), pl.lit(body.name).alias("body"))
        for i, body in enumerate(BODIES)
    ]).select("mass #", "body", "x", "y", "z", "vx", "vy", "vz")
    print("\nFinal simulated barycentric states (AU and AU/day):")
    with pl.Config(tbl_rows=len(BODIES), tbl_cols=8, tbl_formatting="ASCII_FULL"):
        print(final_states)

    days, errors = compare_to_jpl(result, args.checks, args.csv)
    if not args.no_plot:
        import matplotlib.pyplot as plt
        plot_points(result)
        plot_errors(days, errors)
        if args.fit_ellipses:
            plot_ellipses(result)
        if args.fitplot:
            fitplot(result)
        plt.show()
    return result


if __name__ == "__main__":
    main()
