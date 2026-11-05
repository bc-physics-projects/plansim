import numpy as np

import plansim

a = plansim.OrbitalSolver2([100, 2])
out = a.solve([0, 5], [0, 0], [0, 4], [0, np.pi / 2], (0, 10), np.linspace(0, 10, 100))
print(out)
