import random

import numpy as np

import plansim

# a = plansim.OrbitalSolver2([100, 2])
# out1 = a.solve([0, 5], [0, 0], [0, 4], [0, np.pi / 2], (0, 10), np.linspace(0, 10, 100))
# print(out1)

# b = plansim.OrbitalSolver8([10000, 2, 4, 5, 2, 3, 4, 5])
# out2 = b.solve(
#     [0, 5, 10, 100, 25, 34, 17, 87],
#     [0, 0, 0, 0, np.pi, np.pi / 2, 0, 3 * np.pi / 2],
#     [0, 4, 18, 20, 35, 37, 3, 4],
#     [0, np.pi / 2, 0, -np.pi / 2, np.pi / 4, 0, np.pi / 2, np.pi / 2],
#     (0, 10),
#     np.linspace(0, 10, 10000),
# )

c = plansim.OrbitalSolver91(
    [10000] + random.sample(10 * [2, 4, 5, 2, 3, 4, 5, 4, 2], 90)
)
out3 = c.solve(
    [0] + random.sample(10 * [100, 500, 300, 400, 150, 200, 180, 250, 320, 379], 90),
    [0]
    + random.sample(
        10 * [0, 0, 0, np.pi, np.pi / 2, 0, 3 * np.pi / 2, 0, np.pi / 4], 90
    ),
    [0] + random.sample(10 * [3, 4, 4, 18, 20, 35, 37, 3, 4], 90),
    [0]
    + random.sample(
        10
        * [
            np.pi / 2,
            0,
            -np.pi / 2,
            np.pi / 4,
            0,
            np.pi / 2,
            np.pi / 2,
            np.pi / 3,
            np.pi / 6,
        ],
        90,
    ),
    (0, 10),
    # np.linspace(0, 10, 10000),
)
