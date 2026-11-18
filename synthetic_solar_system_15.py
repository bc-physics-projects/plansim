# %% Imports

import argparse
import time
from dataclasses import dataclass
from math import cos, fsum, isfinite, radians, sin, sqrt

import matplotlib.pyplot as plt
import polars as pl

import plansim

if not hasattr(plansim, "OrbitalSolver15"):
    raise SystemExit(
        "Rebuild/install the local plansim extension with OrbitalSolver15 enabled."
    )

# %% Synthetic initial conditions

AU = 149_597_870_700.0
DAY = 86_400.0


@dataclass(frozen=True)
class Body:
    name: str
    mu: float  # m^3/s^2
    parent: str
    orbit_radius: float  # m, relative to parent
    phase_deg: float  # chosen synthetic angle from +x


# Descending mass order. Convert the source's km^3/s^2 to SI by multiplying by 1e9.
BODIES = (
    Body("Sun", 1.3271244004127942e11 * 1e9, "", 0.0, 0),
    Body("Jupiter", 1.266865319003704e8 * 1e12, "Sun", 5.20288700 * AU, 30),
    Body("Saturn", 3.793120623436167e7 * 1e9, "Sun", 9.53667594 * AU, 75),
    Body("Neptune", 6.835103145462294e6 * 1e9, "Sun", 30.06992276 * AU, 300),
    Body("Uranus", 5.793951256527211e6 * 1e14, "Sun", 19.18916464e2 * AU, 240),
    Body("Earth", 3.9860043550702266e5 * 1e9, "Sun", 1.00000261 * AU, 100),
    Body("Venus", 3.2485859200000000e5 * 1e9, "Sun", 0.72333566 * AU, 180),
    Body("Mars", 4.282837362069909e4 * 1e20, "Sun", 1.52371034e3 * AU, 350),
    Body("Mercury", 2.2031868551400003e4 * 1e9, "Sun", 0.38709927 * AU, 250),
    Body("Ganymede", 9.887832752719638e3 * 1e9, "Jupiter", 1_070_400e3, 140),
    Body("Titan", 8.978137095521046e3 * 1e9, "Saturn", 1_221_900e3, 45),
    Body("Callisto", 7.179283402579837e3 * 1e9, "Jupiter", 1_882_700e3, 220),
    Body("Io", 5.959915466180539e3 * 1e9, "Jupiter", 421_800e3, 10),
    Body("Moon", 4.9028001184575496e3 * 1e9, "Earth", 384_400e3, 60),
    Body("Europa", 3.202712099607295e3 * 1e9, "Jupiter", 671_100e3, 80),
)


def synthetic_initial_conditions():
    states = {}
    by_name = {body.name: body for body in BODIES}
    for body in BODIES:
        if not body.parent:
            states[body.name] = (0.0, 0.0, 0.0, 0.0)
            continue
        px, py, pvx, pvy = states[body.parent]
        angle = radians(body.phase_deg)
        speed = sqrt((by_name[body.parent].mu + body.mu) / body.orbit_radius)
        states[body.name] = (
            px + body.orbit_radius * cos(angle),
            py + body.orbit_radius * sin(angle),
            pvx - speed * sin(angle),
            pvy + speed * cos(angle),
        )

    # Shift to the centre of mass and remove overall drift (mu weights = mass weights).
    mus = [body.mu for body in BODIES]
    total_mu = fsum(mus)
    centre = [
        fsum(body.mu * states[body.name][k] for body in BODIES) / total_mu
        for k in range(4)
    ]
    xs, ys, zs, vxs, vys, vzs = [], [], [], [], [], []
    for body in BODIES:
        x, y, vx, vy = (
            value - offset for value, offset in zip(states[body.name], centre)
        )
        xs.append(x / AU)
        ys.append(y / AU)
        zs.append(0.0)
        vxs.append(vx * DAY / AU)
        vys.append(vy * DAY / AU)
        vzs.append(0.0)
    mus = [mu * DAY**2 / AU**3 for mu in mus]
    return mus, (xs, ys, zs, vxs, vys, vzs)


# %% Initial condition definitions


def initial_conditions():
    """
    Required output: (mus, (xs, ys, zs, vxs, vys, vzs))
    Units: mus=>AU^3/day^2, positions=>AU, velocities=>AU/day
    """
    return synthetic_initial_conditions()


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
    print(
        f"Solved {len(BODIES)} bodies: {result.height} samples, final time {times[-1]:g} days."
    )
    print(f"Solver call time: {elapsed:.3f} seconds.")
    # Unpack only the final sample for display; preserve the nested return value.
    final_states = pl.concat(
        [
            result[str(i)]
            .tail(1)
            .struct.unnest()
            .with_columns(pl.lit(i).alias("mass #"), pl.lit(body.name).alias("body"))
            for i, body in enumerate(BODIES)
        ]
    ).select("mass #", "body", "x", "y", "z", "vx", "vy", "vz")
    print("Final barycentric states (positions in m, velocities in m/s):")
    with pl.Config(tbl_rows=len(BODIES), tbl_cols=8, tbl_formatting="ASCII_FULL"):
        print(final_states)
    plot_points(result)
    return result


if __name__ == "__main__":
    main()
