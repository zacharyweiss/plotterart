"""
Zachary Weiss
4 Jan 2021
Rabinovich-Fabrikant Equation
"""
from diffeqpy import de
import matplotlib.pyplot as plt
import numpy as np
import numba
from mpl_toolkits.mplot3d import Axes3D

def f(u, p, t):
    x, y, z = u
    alpha, gamma = p
    return [y * (z - 1 + np.square(x)) + gamma*x, x * (3*z + 1 - np.square(x)) + gamma*y, -2 * z * (alpha + x*y)]


u0 = [-1.0, 0.0, 0.5]
tspan = (0., 100.)
p = [1.1, 0.87]
numba_f = numba.jit(f)
prob = de.ODEProblem(numba_f, u0, tspan, p)
sol = de.solve(prob, saveat=0.01)

ut = np.transpose(sol.u)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(ut[0, :], ut[1, :], ut[2, :])

ax.autoscale()
ax.set_aspect('equal')
plt.grid(False)
plt.axis('off')
plt.tight_layout()
plt.show()
