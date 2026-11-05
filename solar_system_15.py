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

Usage after rebuilding/installing plansim:
    python solar_system_15.py --days 30
    python solar_system_15.py --initial-only
"""

import argparse
from dataclasses import dataclass
from math import atan2, cos, fsum, hypot, isfinite, radians, sin, sqrt

AU = 149_597_870_700.0
DAY = 86_400.0


@dataclass(frozen=True)
class Body:
    name: str
    mu: float                  # m^3/s^2
    parent: str
    orbit_radius: float        # m, relative to parent
    phase_deg: float           # chosen synthetic angle from +x


# Descending mass order. Convert the source's km^3/s^2 to SI by multiplying by 1e9.
BODIES = (
    Body("Sun",      1.3271244004127942e11 * 1e9, "",       0.0,                0),
    Body("Jupiter",  1.266865319003704e8 * 1e9,   "Sun",    5.20288700 * AU,   30),
    Body("Saturn",   3.793120623436167e7 * 1e9,   "Sun",    9.53667594 * AU,   75),
    Body("Neptune",  6.835103145462294e6 * 1e9,   "Sun",   30.06992276 * AU,  300),
    Body("Uranus",   5.793951256527211e6 * 1e9,   "Sun",   19.18916464 * AU,  240),
    Body("Earth",    3.9860043550702266e5 * 1e9,  "Sun",    1.00000261 * AU,  100),
    Body("Venus",    3.2485859200000000e5 * 1e9,  "Sun",    0.72333566 * AU,  180),
    Body("Mars",     4.282837362069909e4 * 1e9,   "Sun",    1.52371034 * AU,  350),
    Body("Mercury",  2.2031868551400003e4 * 1e9,  "Sun",    0.38709927 * AU,  250),
    Body("Ganymede", 9.887832752719638e3 * 1e9,   "Jupiter", 1_070_400e3,     140),
    Body("Titan",    8.978137095521046e3 * 1e9,   "Saturn",  1_221_900e3,      45),
    Body("Callisto", 7.179283402579837e3 * 1e9,   "Jupiter", 1_882_700e3,     220),
    Body("Io",       5.959915466180539e3 * 1e9,   "Jupiter",   421_800e3,      10),
    Body("Moon",     4.9028001184575496e3 * 1e9,  "Earth",     384_400e3,      60),
    Body("Europa",   3.202712099607295e3 * 1e9,   "Jupiter",   671_100e3,      80),
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
        x, y, vx, vy = (value - offset for value, offset in zip(states[body.name], centre))
        radius = hypot(x, y)
        rs.append(radius)
        r_angs.append(atan2(y, x))
        vs.append(hypot(vx, vy))
        v_angs.append(atan2(vy, vx))
        radial_vs.append((x * vx + y * vy) / radius if radius else 0.0)
    return mus, rs, r_angs, vs, v_angs, radial_vs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=30.0)
    parser.add_argument("--initial-only", action="store_true", help="Print initial data without solving")
    args = parser.parse_args()
    if not isfinite(args.days) or args.days <= 0:
        parser.error("--days must be finite and positive")

    mus, rs, r_angs, vs, v_angs, radial_vs = initial_conditions()
    print("Synthetic coplanar initial conditions; barycentric radii and velocities.")
    print(f"{'Body':<10} {'mu [m^3/s^2]':>15} {'r [m]':>15} {'angle [rad]':>12} "
          f"{'speed [m/s]':>13} {'v angle [rad]':>14} {'dr/dt [m/s]':>13}")
    for i, body in enumerate(BODIES):
        print(f"{body.name:<10} {mus[i]:15.8e} {rs[i]:15.8e} {r_angs[i]:12.6f} "
              f"{vs[i]:13.6f} {v_angs[i]:14.6f} {radial_vs[i]:13.6f}")
    if args.initial_only:
        return

    import plansim

    if not hasattr(plansim, "OrbitalSolver15"):
        raise SystemExit("Rebuild/install the local plansim extension with OrbitalSolver15 enabled.")
    solver = plansim.OrbitalSolver15(mus)
    result = solver.solve(rs, r_angs, vs, v_angs, (0.0, args.days * DAY))
    times = result.get_t()
    print(f"Solved {len(BODIES)} bodies: {len(times)} samples, final time {times[-1] / DAY:g} days.")
    return result


if __name__ == "__main__":
    main()
