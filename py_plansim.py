"""Pure-Python equivalent of src/system.rs.

The flat state stores x, y, z, vx, vy, vz consecutively for each body.
Use metres and seconds, with gravitational parameters mu = GM in m^3/s^2.
"""

import numpy as np
from CyRK import nbsolve_ivp
from numba import njit


@njit(cache=True, fastmath=True, nogil=True)
def system(t: float, y: np.ndarray, mus: np.ndarray) -> np.ndarray:
    n = len(mus)

    if len(y) != 6 * n:
        raise ValueError("y must have length 6 * len(mus)")

    dy = np.empty_like(y)
    for i in range(n):
        acc = [0.0, 0.0, 0.0]
        for j in range(n):
            if i != j:
                r_vec = [y[j * 6 + k] - y[i * 6 + k] for k in range(3)]
                r = np.sqrt(r_vec[0] ** 2 + r_vec[1] ** 2 + r_vec[2] ** 2)
                scale = mus[j] / r**3
                for k in range(3):
                    acc[k] += r_vec[k] * scale

        for k in range(3):
            dy[i * 6 + k] = y[i * 6 + k + 3]
            dy[i * 6 + k + 3] = acc[k]

    return dy


class OrbitalSolver:
    def __init__(self, mus: list[float]):
        self.mus = np.asarray(mus, dtype=np.float64)
        self.n = len(mus)
        self.y0 = np.zeros(6 * self.n, dtype=np.float64)

    def solve(
        self,
        rs: list[float],
        r_angs: list[float],
        vs: list[float],
        v_angs: list[float],
        t_span: tuple[float, float],
        tspace: float,
    ) -> tuple[np.ndarray, np.ndarray]:
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

        sol = nbsolve_ivp(
            system,
            t_span,
            self.y0,
            t_eval=np.arange(*t_span, tspace),
            rk_method=2,  # "DOP853"
            args=(self.mus,),
            warnings=False,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        return (sol.t, sol.y)
