from diffeqpy import de
import matplotlib.pyplot as plt
import numpy as np
import numba


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

plt.plot(sol.t, sol.u)
plt.show()
