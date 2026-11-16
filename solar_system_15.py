"""Planar example with the Sun, eight planets, and six most massive moons.

Units: metres, seconds, radians; mu = GM in m^3/s^2 (do not multiply by G).
"Largest" means most massive, including the Sun. Initial orbits are circular
approximations with synthetic phases, not an ephemeris for a particular date.
All inclinations and eccentricities are ignored. Radius means orbital distance,
not the body's physical radius. Moons inherit their parent's position/velocity.

Sources (GM values are individual bodies, not planet-plus-moon systems):
https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc
https://ssd.jpl.nasa.gov/planets/approx_pos.html  (orbital scales)
https://ssd.jpl.nasa.gov/sats/elem/  (moon orbital scales)

The solver returns a Polars DataFrame: t is time in seconds, and columns
"0" through "14" follow BODIES order. Each body column contains the Struct
fields x, y, z, vx, vy, vz. For example, result["5"].struct.field("x")
gives Earth's x coordinates. main() displays all bodies' sampled x/y positions
in astronomical units, then returns this nested DataFrame when the plot closes.

Usage after rebuilding/installing plansim:
    python solar_system_15.py --days 30
    python solar_system_15.py --days 30 --tspace 60
    python solar_system_15.py --initial-only
"""

import argparse
import time
from dataclasses import dataclass
from math import atan2, cos, fsum, hypot, isfinite, radians, sin, sqrt

import numpy as np
from ellipse import LsqEllipse

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
    Body("Jupiter", 1.266865319003704e8 * 1e9, "Sun", 5.20288700 * AU, 30),
    Body("Saturn", 3.793120623436167e7 * 1e9, "Sun", 9.53667594 * AU, 75),
    Body("Neptune", 6.835103145462294e6 * 1e9, "Sun", 30.06992276 * AU, 300),
    Body("Uranus", 5.793951256527211e6 * 1e9, "Sun", 19.18916464 * AU, 240),
    Body("Earth", 3.9860043550702266e5 * 1e9, "Sun", 1.00000261 * AU, 100),
    Body("Venus", 3.2485859200000000e5 * 1e9, "Sun", 0.72333566 * AU, 180),
    Body("Mars", 4.282837362069909e4 * 1e9, "Sun", 1.52371034 * AU, 350),
    Body("Mercury", 2.2031868551400003e4 * 1e9, "Sun", 0.38709927 * AU, 250),
    Body("Ganymede", 9.887832752719638e3 * 1e9, "Jupiter", 1_070_400e3, 140),
    Body("Titan", 8.978137095521046e3 * 1e9, "Saturn", 1_221_900e3, 45),
    Body("Callisto", 7.179283402579837e3 * 1e9, "Jupiter", 1_882_700e3, 220),
    Body("Io", 5.959915466180539e3 * 1e9, "Jupiter", 421_800e3, 10),
    Body("Moon", 4.9028001184575496e3 * 1e9, "Earth", 384_400e3, 60),
    Body("Europa", 3.202712099607295e3 * 1e9, "Jupiter", 671_100e3, 80),
)


def initial_conditions():
    """Return solver arrays and radial speeds in the model's barycentric frame.

    vs is TOTAL speed; v_angs is the velocity direction from +x, as required by
    the Rust solver. radial_vs = dr/dt, positive outward, is separate.
    """
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
    rs, r_angs, vs, v_angs, radial_vs = [], [], [], [], []
    for body in BODIES:
        x, y, vx, vy = (
            value - offset for value, offset in zip(states[body.name], centre)
        )
        radius = hypot(x, y)
        rs.append(radius)
        r_angs.append(atan2(y, x))
        vs.append(hypot(vx, vy))
        v_angs.append(atan2(vy, vx))
        radial_vs.append((x * vx + y * vy) / radius if radius else 0.0)
    return mus, rs, r_angs, vs, v_angs, radial_vs


def ellipse(x, y, A, B):
    """Return the ellipse equation value for a point (x, y) in AU."""
    # Use the Sun's position as the origin; the Sun is at (0, 0) in this model.
    # The semi-major axis is the average distance of the planet from the Sun.
    # The semi-minor axis is calculated based on the eccentricity of the orbit.
    # For simplicity, we assume circular orbits (eccentricity = 0), so a = b.
    return x**2 + y**2


def plot_points(result):
    """Display every sampled position of the Sun, planets, and moons in x/y."""
    import matplotlib.pyplot as plt

    cx = cy = 0.0

    for i, body in enumerate(BODIES):
        state = result[str(i)]
        x = state.struct.field("x").to_numpy() / AU
        y = state.struct.field("y").to_numpy() / AU
        mu = body.mu
        cx += mu * x
        cy += mu * y
    cx /= fsum(body.mu for body in BODIES)
    cy /= fsum(body.mu for body in BODIES)

    print(cx.max(), cx.min(), cy.max(), cy.min())
    print(cx[0], cy[0], cx[-1], cy[-1])

    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    colors = plt.get_cmap("tab20").colors
    for i, body in enumerate(BODIES):
        state = result[str(i)]
        x = state.struct.field("x").to_numpy() / AU
        y = state.struct.field("y").to_numpy() / AU
        ax.plot(
            x - cx,
            y - cy,
            linestyle="None",
            marker=".",
            markersize=2,
            color=colors[i],
            label=body.name,
            rasterized=True,
        )

    ax.set(
        xlabel="Barycentric x [AU]",
        ylabel="Barycentric y [AU]",
        title=f"Sun, planet, and moon positions over {result['t'][-1] / DAY:g} days",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), markerscale=4)
    plt.show()
    return fig


def fit_ellipse_parameters(x, y):
    """Fit in centered, scaled coordinates to reduce numerical cancellation."""
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


def plot_ellipses(result):
    """Plot fitted planetary ellipses and print their least-squares errors."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    colors = plt.get_cmap("tab20").colors

    print("Planet ellipse fit errors (normalized ellipse-equation residuals):")
    print(f"{'Planet':<10} {'MSE':>14} {'RMSE':>14}")

    # BODIES[1:9] are the eight planets; omit the Sun and moons.
    for i, body in enumerate(BODIES[1:9], start=1):
        state = result[str(i)]
        x = state.struct.field("x").to_numpy() / AU
        y = state.struct.field("y").to_numpy() / AU

        try:
            (xc, yc), a, b, theta = fit_ellipse_parameters(x, y)
        except (ValueError, IndexError, np.linalg.LinAlgError) as exc:
            print(f"{body.name:<10} Fit unavailable: {exc}")
            continue

        # Evaluate the fitted ellipse equation at each sampled position.
        dx = x - xc
        dy = y - yc
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x_rot = dx * cos_theta + dy * sin_theta
        y_rot = -dx * sin_theta + dy * cos_theta
        residuals = (x_rot / a) ** 2 + (y_rot / b) ** 2 - 1
        mse = float(np.mean(residuals**2))
        rmse = sqrt(mse)
        print(f"{body.name:<10} {mse:14.6e} {rmse:14.6e}")

        t = np.linspace(0, 2 * np.pi, 100)
        ellipse_x = xc + a * np.cos(t) * cos_theta - b * np.sin(t) * sin_theta
        ellipse_y = yc + a * np.cos(t) * sin_theta + b * np.sin(t) * cos_theta
        ax.plot(
            ellipse_x,
            ellipse_y,
            linestyle="-",
            color=colors[i],
            label=f"{body.name} fit",
            alpha=0.7,
        )

    ax.set(
        xlabel="Barycentric x [AU]",
        ylabel="Barycentric y [AU]",
        title=f"Fitted planetary ellipses over {result['t'][-1] / DAY:g} days",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.show()
    return fig


def fitplot(result):
    """Compare each planet's raw barycentric samples with its fitted ellipse."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), layout="constrained")
    colors = plt.get_cmap("tab20").colors
    phase = np.linspace(0, 2 * np.pi, 500)
    for i, (body, ax) in enumerate(zip(BODIES[1:9], axes.flat), start=1):
        state = result[str(i)]
        x = state.struct.field("x").to_numpy() / AU
        y = state.struct.field("y").to_numpy() / AU
        ax.plot(
            x,
            y,
            linestyle="None",
            marker=".",
            markersize=3,
            color=colors[i],
            label="Raw samples",
            rasterized=True,
            zorder=3,
        )
        try:
            (xc, yc), a, b, theta = fit_ellipse_parameters(x, y)
            ct, st = np.cos(theta), np.sin(theta)
            ellipse_x = xc + a * np.cos(phase) * ct - b * np.sin(phase) * st
            ellipse_y = yc + a * np.cos(phase) * st + b * np.sin(phase) * ct
            ax.plot(
                ellipse_x,
                ellipse_y,
                "--",
                color="black",
                linewidth=1.2,
                label="Fitted ellipse",
            )
        except (ValueError, IndexError, np.linalg.LinAlgError) as exc:
            ax.text(
                0.03,
                0.97,
                f"Fit unavailable:\n{exc}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                wrap=True,
            )
        ax.set(
            title=body.name, xlabel="Barycentric x [AU]", ylabel="Barycentric y [AU]"
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, markerscale=3)
    fig.suptitle(
        f"Planet samples and ellipse fits — {result['t'][-1] / DAY:g} days\n"
        "Short orbital arcs may poorly constrain the full ellipse."
    )
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=30.0)
    parser.add_argument(
        "--tspace",
        type=float,
        default=1.0,
        help="Time between samples in seconds (default: 1)",
    )
    parser.add_argument(
        "--initial-only", action="store_true", help="Print initial data without solving"
    )
    args = parser.parse_args()
    if not isfinite(args.days) or args.days <= 0:
        parser.error("--days must be finite and positive")
    if not isfinite(args.tspace) or args.tspace <= 0:
        parser.error("--tspace must be finite and positive")

    mus, rs, r_angs, vs, v_angs, radial_vs = initial_conditions()
    print("Synthetic coplanar initial conditions; barycentric radii and velocities.")
    print(
        f"{'Body':<10} {'mu [m^3/s^2]':>15} {'r [m]':>15} {'angle [rad]':>12} "
        f"{'speed [m/s]':>13} {'v angle [rad]':>14} {'dr/dt [m/s]':>13}"
    )
    for i, body in enumerate(BODIES):
        print(
            f"{body.name:<10} {mus[i]:15.8e} {rs[i]:15.8e} {r_angs[i]:12.6f} "
            f"{vs[i]:13.6f} {v_angs[i]:14.6f} {radial_vs[i]:13.6f}"
        )
    if args.initial_only:
        return

    import polars as pl

    import plansim

    if not hasattr(plansim, "OrbitalSolver15"):
        raise SystemExit(
            "Rebuild/install the local plansim extension with OrbitalSolver15 enabled."
        )
    solver = plansim.OrbitalSolver15(mus)
    start = time.perf_counter()
    result = solver.solve(rs, r_angs, vs, v_angs, (0.0, args.days * DAY), args.tspace)
    elapsed = time.perf_counter() - start
    if not isinstance(result, pl.DataFrame):
        raise SystemExit("Rebuild/install plansim for the Polars DataFrame API.")
    times = result["t"]
    print(
        f"Solved {len(BODIES)} bodies: {result.height} samples, final time {times[-1] / DAY:g} days."
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
    # plot_ellipses(result)
    fitplot(result)
    return result


if __name__ == "__main__":
    main()
