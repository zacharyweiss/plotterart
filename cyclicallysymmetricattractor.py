"""
Zachary Weiss
4 Jan 2021
Thomas' Cyclically Symmetric Attractor
"""
from diffeqpy import de
import matplotlib.pyplot as plt
import numpy as np
import numba
from mpl_toolkits.mplot3d import Axes3D


def f(u, p, t):
    x, y, z = u
    beta = p
    return [np.sin(y)-beta*x, np.sin(z)-beta*y, np.sin(x)-beta*z]


u0 = [-1.0, 0.0, 0.5]
tspan = (0., 1000.)
p = 0.32
numba_f = numba.jit(f)
prob = de.ODEProblem(numba_f, u0, tspan, p)
sol = de.solve(prob, saveat=0.01)

ut = np.transpose(sol.u)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(ut[0, :], ut[1, :], ut[2, :])

ax.autoscale()
# ax.set_aspect('equal')
plt.grid(False)
plt.axis('off')
plt.tight_layout()
plt.show()
