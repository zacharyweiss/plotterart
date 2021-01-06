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
from matplotlib import animation


def f(u, p, t):
    x, y, z = u
    beta = p
    return [np.sin(y)-beta*x, np.sin(z)-beta*y, np.sin(x)-beta*z]


u0 = [-0.5, 0.5, 4.]
tspan = (0., 1000.)
# b>1 origin only stable eq pt
#  =1 pitchfork bifurcation
# ~=0.32899 Hopf Bifurcation (to stable limit cycle)
# ~=0.208186 period doubling -> chaotic
# lim->0 wanders space w/o dissipation
p = 0.1998
numba_f = numba.jit(f)
prob = de.ODEProblem(numba_f, u0, tspan, p)
sol = de.solve(prob, saveat=0.01)

ut = np.transpose(sol.u)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def init():
    ax.plot(ut[0, :], ut[1, :], ut[2, :])
    ax.autoscale()
    # ax.set_aspect('equal')
    plt.grid(False)
    plt.axis('off')
    plt.tight_layout()
    return fig,


def animate(i):
    ax.view_init(elev=10., azim=i)
    return fig,


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=20, blit=True)
anim.save('animationtest.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

