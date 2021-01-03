"""
Zachary Weiss
30 Dec 2020
Lorenz Equation
"""
from diffeqpy import de
import matplotlib.pyplot as plt
import numpy as np
import numba
from mpl_toolkits.mplot3d import Axes3D


def f(u, p, t):
    x, y, z = u
    sigma, rho, beta = p
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


u0 = [1.0, 0.0, 0.0]
tspan = (0., 100.)
p = [10.0, 28.0, 8 / 3]
numba_f = numba.jit(f)
prob = de.ODEProblem(numba_f, u0, tspan, p)
sol = de.solve(prob, saveat=0.01)

ut = np.transpose(sol.u)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(ut[0, :], ut[1, :], ut[2, :])
plt.show()