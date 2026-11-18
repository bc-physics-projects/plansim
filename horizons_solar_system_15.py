# %% Imports

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
from math import isfinite

import numpy as np
from astropy.time import Time
from astroquery.jplhorizons import Horizons


AU_KM = 149_597_870.7
DAY_S = 86_400.0
# Earliest shared epoch verified for these 15 Horizons targets. Neptune's
# planet-center ephemeris sets this lower bound. Use TDB directly: UTC was
# not defined in 1800, and the coverage boundary is specified in dynamical time.
START_TDB = "1800-01-02 00:00:00"
# Neptune also has the earliest upper limit of the planet-center targets.
# Default to the shared window with a one-day margin at its upper boundary.
COMPARISON_END_TDB = "2199-12-30 00:00:00"
DEFAULT_DAYS = float(
    Time(COMPARISON_END_TDB, format="iso", scale="tdb").jd
    - Time(START_TDB, format="iso", scale="tdb").jd
    - 1.0
)

# Each GM is for the *individual* body (not its system barycenter), in km^3/s^2.
# Source: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc
@dataclass(frozen=True)
class Body:
    name: str
    horizons_id: str
    gm_km3_s2: float


# Preserve the original script's exact 15-body order.
# Planet-center states match the individual planet GMs used by the solver.
# Positions remain relative to the solar-system barycenter; the targets are
# individual bodies, not planetary-system barycenters.
BODIES = (
    Body("Sun",      "10",  1.3271244004127942e11),
    Body("Jupiter", "599", 1.266865319003704e8),
    Body("Saturn", "699", 3.793120623436167e7),
    Body("Neptune", "899", 6.835103145462294e6),
    Body("Uranus", "799", 5.793951256527211e6),
    Body("Earth", "399", 3.9860043550702266e5),
    Body("Venus", "299", 3.2485859200000000e5),
    Body("Mars", "499", 4.282837362069909e4),
    Body("Mercury", "199", 2.2031868551400003e4),
    Body("Ganymede", "503", 9.887832752719638e3),
    Body("Titan",    "606", 8.978137095521046e3),
    Body("Callisto", "504", 7.179283402579837e3),
    Body("Io",       "501", 5.959915466180539e3),
    Body("Moon",     "301", 4.9028001184575496e3),
    Body("Europa",   "502", 3.202712099607295e3),
)
COMPARISON_BODIES = BODIES[:9]  # Sun and planets only; moons are initialized and integrated.
FIELDS = ("x", "y", "z", "vx", "vy", "vz")

def start_jd():
    """Return the historical start epoch as a TDB Julian Date for Horizons."""
    return float(Time(START_TDB, format="iso", scale="tdb").jd)


def jpl_states(julian_dates, bodies=BODIES):
    """Retrieve barycentric states at TDB Julian Dates.

    Returns ndarray shape (number_of_epochs, len(bodies), 6), with positions in AU
    and velocities in AU/day. Each body is queried independently, then kept
    in the supplied bodies order. This works for moons without adding parent state vectors.
    """
    jds = [float(jd) for jd in julian_dates]
    if not jds:
        raise ValueError("At least one epoch is required")

    states = np.empty((len(jds), len(bodies), 6), dtype=float)
    for i, body in enumerate(bodies):
        print(f"Horizons: {body.name:<10} ({i + 1}/{len(bodies)})", flush=True)
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


def final_reference_if_available(days):
    """Check endpoint coverage before integration; return None if unavailable."""
    epoch = start_jd() + days
    if epoch > float(Time(COMPARISON_END_TDB, format="iso", scale="tdb").jd):
        print(f"\nSkipping final comparison: the requested endpoint exceeds "
              f"Neptune's coverage limit ({COMPARISON_END_TDB} TDB).")
        print("The simulation will still run; only calculated positions will be plotted.")
        return None
    print("\nChecking Horizons coverage at the requested end date ...")
    try:
        return jpl_states([epoch], bodies=COMPARISON_BODIES)[0]
    except ValueError as exc:
        if "No ephemeris for target" not in str(exc):
            raise
        final_date = Time(epoch, format="jd", scale="tdb").iso
        print(f"Skipping final comparison: {final_date} TDB is outside "
              f"Horizons coverage.\n{exc}")
        print("The simulation will still run; only calculated positions will be plotted.")
        return None


def compare_final_to_horizons(result, csv_path=None, reference=None):
    """Compare the Sun and planets at the last solver sample with Horizons at the same TDB epoch.

    Errors include integration error and differences between this 15-body model
    and Horizons' ephemeris model. Reference states never enter the integration.
    Signed component errors are simulation minus Horizons.
    Relative errors are percentages of the corresponding Horizons vector norms.
    A zero reference norm gives NaN.
    """
    if result.is_empty():
        raise ValueError("Cannot compare an empty solver result")
    final_day = float(result["t"][-1])
    if not isfinite(final_day):
        raise ValueError("The final solver time must be finite")
    epoch = start_jd() + final_day
    computed = np.array([
        [result[str(i)].struct.field(field)[-1] for field in FIELDS]
        for i in range(len(COMPARISON_BODIES))
    ], dtype=float)
    if not np.isfinite(computed).all():
        raise ValueError("The final solver states must be finite")
    if reference is None:
        reference = jpl_states([epoch], bodies=COMPARISON_BODIES)[0]
    delta = computed - reference
    position_delta = delta[:, :3] * AU_KM
    velocity_delta = delta[:, 3:] * AU_KM / DAY_S
    position_errors = np.linalg.norm(position_delta, axis=1)
    velocity_errors = np.linalg.norm(velocity_delta, axis=1)
    reference_position_km = np.linalg.norm(reference[:, :3], axis=1) * AU_KM
    reference_velocity_km_s = np.linalg.norm(reference[:, 3:], axis=1) * AU_KM / DAY_S
    position_errors_pct = np.divide(
        100 * position_errors, reference_position_km,
        out=np.full_like(position_errors, np.nan), where=reference_position_km > 0,
    )
    velocity_errors_pct = np.divide(
        100 * velocity_errors, reference_velocity_km_s,
        out=np.full_like(velocity_errors, np.nan), where=reference_velocity_km_s > 0,
    )
    errors = pl.DataFrame({
        "body": [body.name for body in COMPARISON_BODIES],
        "time_days": [final_day] * len(COMPARISON_BODIES),
        "epoch_jd_tdb": [epoch] * len(COMPARISON_BODIES),
        **{f"horizons_{axis}_au": reference[:, i] for i, axis in enumerate("xyz")},
        **{f"d{axis}_km": position_delta[:, i] for i, axis in enumerate("xyz")},
        **{f"dv{axis}_km_s": velocity_delta[:, i] for i, axis in enumerate("xyz")},
        "position_error_km": position_errors,
        "velocity_error_km_s": velocity_errors,
        "position_error_pct": position_errors_pct,
        "velocity_error_pct": velocity_errors_pct,
    })
    print(f"\nHorizons comparison at t={final_day:.9g} days (JD {epoch:.12f} TDB)")
    print(
        f"{'Body':<12} {'Position error [km]':>21} {'Velocity error [km/s]':>24}"
        f" {'Position error [%]':>21} {'Velocity error [%]':>21}"
    )
    for body, position, velocity, position_pct, velocity_pct in zip(
        COMPARISON_BODIES, position_errors, velocity_errors,
        position_errors_pct, velocity_errors_pct,
    ):
        print(
            f"{body.name:<12} {position:21.9e} {velocity:24.9e}"
            f" {position_pct:21.9e} {velocity_pct:21.9e}"
        )
    print("Errors include numerical integration and physical-model differences.")
    if csv_path is not None:
        errors.write_csv(csv_path)
        print(f"Final error details saved to {csv_path}")
    return errors


def plot_points(result, final_errors=None):
    """Plot XY trajectories and matching-color calculated/Horizons endpoints."""
    from matplotlib.lines import Line2D

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
        if i < len(COMPARISON_BODIES):
            ax.plot(
                x[-1], y[-1], linestyle="none", marker="o", markersize=6,
                color=colors[i], zorder=4,
            )
        if final_errors is not None and i < len(COMPARISON_BODIES):
            ax.plot(
                final_errors["horizons_x_au"][i],
                final_errors["horizons_y_au"][i],
                linestyle="none", marker="x", markersize=10,
                markeredgewidth=2, color=colors[i], zorder=5,
            )

    ax.set(
        xlabel="Barycentric x [AU]",
        ylabel="Barycentric y [AU]",
        title=f"Sun, planet, and moon positions over {result['t'][-1]:g} days",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="black", linestyle="none", marker="o",
                          markersize=6))
    labels.append("Calculated final position")
    if final_errors is not None:
        handles.append(Line2D([], [], color="black", linestyle="none", marker="x",
                              markersize=10, markeredgewidth=2))
        labels.append("Horizons final position")
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days", type=float, default=DEFAULT_DAYS,
        help=f"Simulation duration in days (default: {DEFAULT_DAYS:g}, "
             "shared planet coverage with a one-day margin)",
    )
    parser.add_argument(
        "--tspace",
        type=float,
        default=1.0,
        help="Time between samples in days (default: 1)",
    )
    parser.add_argument(
        "--initial-only", action="store_true", help="Print initial data without solving"
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip the orbit plot")
    parser.add_argument(
        "--skip-comparison", action="store_true",
        help="Skip the final Horizons query (useful beyond ephemeris coverage)",
    )
    parser.add_argument(
        "--error-csv", help="Save final Horizons errors, including signed components, to CSV"
    )
    args = parser.parse_args()
    if not isfinite(args.days) or args.days <= 0:
        parser.error("--days must be finite and positive")
    if not isfinite(args.tspace) or args.tspace <= 0:
        parser.error("--tspace must be finite and positive")

    print(f"Simulation start: {START_TDB} TDB")
    mus, init_data = initial_conditions()
    if args.initial_only:
        print(f"Initial conditions at {START_TDB} TDB (AU, AU/day):")
        for body, state in zip(BODIES, np.asarray(init_data).T):
            print(f"{body.name:<12} " + " ".join(f"{value:.12e}" for value in state))
        return mus, init_data

    reference = None
    if not args.skip_comparison:
        reference = final_reference_if_available(args.days)
        if reference is None and args.error_csv:
            print(f"No error CSV will be written to {args.error_csv} for this run.")

    solver = plansim.OrbitalSolver15(mus, kill_sigint=True)
    start = time.perf_counter()
    result = solver.solve(*init_data, (0.0, args.days), args.tspace)
    elapsed = time.perf_counter() - start
    if not isinstance(result, pl.DataFrame):
        raise SystemExit("Rebuild/install plansim for the Polars DataFrame API.")
    times = result["t"]
    print(f"Solver call time: {elapsed:.3f} seconds.")
    print(f"Final time: {times[-1]:g} days.")
    final_errors = None
    if reference is not None:
        try:
            # Reuse the coverage query only when the solver reached that exact
            # time. Otherwise query the actual last sample, avoiding date mismatch.
            final_errors = compare_final_to_horizons(
                result, args.error_csv,
                reference=reference if float(times[-1]) == args.days else None,
            )
        except ValueError as exc:
            if "No ephemeris for target" not in str(exc):
                raise
            final_date = Time(start_jd() + float(times[-1]), format="jd", scale="tdb").iso
            print(f"\nFinal Horizons comparison unavailable at {final_date} TDB: {exc}")
            print("The simulation completed, but its endpoint is outside Horizons coverage "
                  "for at least one body. Plotting calculated positions only. "
                  "Use a shorter --days value for a reference comparison.")
            if args.error_csv:
                print(f"No error CSV written to {args.error_csv} for this run.")
    if not args.no_plot:
        plot_points(result, final_errors)
    return result


if __name__ == "__main__":
    main()
