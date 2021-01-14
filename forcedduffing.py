"""
Zachary Weiss
30 Dec 2020
Forced Duffing Equation
"""
from diffeqpy import de
import matplotlib.pyplot as plt
import numpy as np
import numba
from mpl_toolkits.mplot3d import Axes3D


def f(u, p, t):
    # assumes omega_0 = 1
    x, y, z = u
    alpha, beta, delta, gamma, omega, phi = p
    return [y, x-beta*pow(x, 3)-delta*y+gamma*np.cos(z + phi), gamma]


u0 = [1.0, 0.0, 0.]
tspan = (0., 1000.)
# p = [1., 5., 0.02, 8., 0.5, 0.]
# p = [-1., 1., 0.3, 0.5, 1.2, 0.]
p = [-1., 1., 0.25, 0.3, 1., 0.]
numba_f = numba.jit(f)
prob = de.ODEProblem(numba_f, u0, tspan, p)
sol = de.solve(prob, saveat=0.01)

ut = np.transpose(sol.u)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(ut[0, :], ut[1, :])
# ax.autoscale()
# plt.grid(False)
# plt.axis('off')
# plt.tight_layout()
# ax.set_aspect('equal')
# plt.show()

sfactor = 1

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot(ut[0, :]+sfactor*np.cos(ut[2, :]), sfactor*np.sin(ut[2, :]), ut[1, :])
ax.autoscale()
plt.grid(False)
plt.axis('off')
plt.tight_layout()
plt.show()
