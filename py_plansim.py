"""Pure-Python equivalent of src/system.rs.

State rows are x, y, z, vx, vy, vz; each column represents one body.
Use metres and seconds, with gravitational parameters mu = GM in m^3/s^2.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class NBodies:
    mus: list[float]

    def system(self, t: float, y: list[float]) -> list[float]:
        n = len(self.mus)

        if len(y) != 6 * n:
            raise ValueError("y must have shape (6, len(mus))")

        dy = [0.0] * (6 * n)
        for i in range(n):
            acc = [0.0, 0.0, 0.0]
            for j in range(n):
                if i != j:
                    r_vec = [y[j * 6 + k] - y[i * 6 + k] for k in range(3)]
                    r = np.sqrt(sum(component * component for component in r_vec))
                    scale = self.mus[j] / r**3
                    for k in range(3):
                        acc[k] += r_vec[k] * scale

            for k in range(3):
                dy[i * 6 + k] = y[i * 6 + k + 3]
                dy[i * 6 + k + 3] = acc[k]

        return dy


class OrbitalSolver:
    def __init__(self, mus: list[float]):
        self.mus = mus
        self.n = len(mus)
        self.y0 = [0.0] * (6 * self.n)
        self.system = NBodies(mus)

    def solve(
        self,
        rs: list[float],
        r_angs: list[float],
        vs: list[float],
        v_angs: list[float],
        t_span: tuple[float, float],
        tspace: float,
    ) -> tuple[list[float], list[list[float]]]:
        if (
            len(rs) != self.n
            or len(r_angs) != self.n
            or len(vs) != self.n
            or len(v_angs) != self.n
        ):
            raise ValueError("Input lists must have length equal to number of bodies")

        for i in range(self.n):
            self.y0[6 * i + 0] = rs[i] * np.cos(r_angs[i])
            self.y0[6 * i + 1] = rs[i] * np.sin(r_angs[i])
            self.y0[6 * i + 2] = 0.0
            self.y0[6 * i + 3] = vs[i] * np.cos(v_angs[i])
            self.y0[6 * i + 4] = vs[i] * np.sin(v_angs[i])
            self.y0[6 * i + 5] = 0.0

        # Here you would implement the numerical integration method (e.g., Runge-Kutta)
        # For simplicity, we will just return the initial state repeated for each time step
        sol = solve_ivp(
            self.system.system,
            t_span,
            self.y0,
            t_eval=np.arange(*t_span, tspace),
            method="DOP853",
        )
        return (sol.t, sol.y)
