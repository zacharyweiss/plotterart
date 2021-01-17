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

# p = [1., 5., 0.02, 8., 0.5, 0.]
# p = [-1., 1., 0.3, 0.5, 1.2, 0.]
# u0 = [1.0, 0.0, 0.]
# u0 = [0., 0.05, 0.]
p = [-1., 1., 0.05, 0.3, 2. * np.pi * 0.2, 0.]
tspan = (0., 5000.)


def f(u, p, t):
    # assumes omega_0 = 1
    x, y, z = u
    alpha, beta, delta, gamma, omega, phi = p
    return [y, x - beta * np.power(x, 3) - delta * y + gamma * np.cos(z + phi), gamma]


def solver(u0, p, tspan):
    numba_f = numba.jit(f)
    prob = de.ODEProblem(numba_f, u0, tspan, p)
    sol = de.solve(prob, saveat=0.01)
    # ut = np.transpose(sol.u)
    ut = np.array(sol.u)

    # change of variables, to account for omega being modular/periodic
    # sfactor = 1
    # ut[0, :] = ut[0, :] + sfactor * np.cos(ut[2, :])
    # ut[2, :] = sfactor * np.sin(ut[2, :])
    return ut


def poincareMap(x0, y0):
    # https://stackoverflow.com/questions/53792164/how-to-implement-a-method-to-generate-poincar%C3%A9-sections-for-a-non-linear-system
    px, py = [], []
    u0 = [x0, y0, 0.]

    u = solver(u0, p, tspan)
    # u0 = u[-1]
    u = np.mod(u + np.pi, 2 * np.pi) - np.pi
    x, y, z = np.transpose(u)

    for k in range(len(z) - 1):
        if z[k] <= 0 <= z[k + 1] and z[k + 1] - z[k] < np.pi:
            # find a more exact intersection location by linear interpolation
            s = -z[k] / (z[k + 1] - z[k])  # 0 = z[k] + s*(z[k+1]-z[k])
            rx, ry = (1 - s) * x[k] + s * x[k + 1], (1 - s) * y[k] + s * y[k + 1]
            px.append(rx)
            py.append(ry)
    return px, py

fig = plt.figure()
ax = fig.add_subplot(111)
N = 100
grid = np.zeros([N, N], dtype=int)
for i in range(N):
    for j in range(N):
        if grid[i, j] > 0: continue;
        x0, y0 = float((2 * i + 1) * np.pi / N - np.pi), float((2 * j + 1) * np.pi / N - np.pi)
        px, py = poincareMap(x0, y0)
        for rx, ry in zip(px, py):
            m, n = int((rx + np.pi) * N / (2 * np.pi)), int((ry + np.pi) * N / (2 * np.pi))
            grid[m, n] = 1

    ax.plot(px, py, '.', ms=2)
ax.autoscale()
plt.axis('off')
plt.tight_layout()
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(ut[0, :], ut[1, :])
# ax.autoscale()
# plt.grid(False)
# plt.axis('off')
# plt.tight_layout()
# plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(ut[0, :], ut[2, :], ut[1, :])
# ax.autoscale()
# plt.grid(False)
# plt.axis('off')
# plt.tight_layout()
# plt.show()
