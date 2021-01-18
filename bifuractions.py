"""
Zachary Weiss
30 Dec 2020
Bifurcation and Lyapunov Exponent
"""
import numpy as np
import matplotlib.pyplot as plt
import math


def diffeq(r, x):
    # logistic
    return r * x * (1 - x)


#def diffeq(k, x):
#    # circle map, for arnold tongue
#    omega = 1/3
#    return x + omega + (k/(2*math.pi))*np.sin(2*math.pi*x)
#diffeq = np.vectorize(circlemap)

def plot_system(r, x0, n, ax=None):
    # plot the fx and the y=x diagonal line
    t = np.linspace(0, 1)
    ax.plot(t, diffeq(r, t), 'k', lw=2)
    ax.plot([0, 1], [0, 1], 'k', lw=2)

    # recursively apply y = f(x)
    # plot (x, x) -> (x, y)
    # plot (x, y) -> (y, y)
    x = x0
    for i in range(n):
        y = diffeq(r, x)
        # plot the lines
        ax.plot([x, x], [x, y], 'k', lw=2)
        ax.plot([x, y], [y, y], 'k', lw=2)
        # plot the positions with increasing opacity
        ax.plot([x], [y], 'ok', ms=10, alpha=(i+1)/n)
        x = y

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"$r={r:.1f}, \, x_0={x0:.1f}$")


n = 10000
r = np.linspace(0, 4*math.pi, n)
iterations = 1000
last = 100
x = 1e-5 * np.ones(n)
lyap = np.zeros(n)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))
for i in range(iterations):
    x = diffeq(r, x)
    # compute partial sum of lyapunov exponent
    lyap += np.log(abs(r - 2 * r * x))
    # display bifurcation diagram
    if i >= (iterations-last):
        ax1.plot(np.mod(r,1), np.mod(x,2*math.pi), ',k', alpha=0.25)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_title("Bifurcation diagram")

# display lyap exponent
# horizontal line
ax2.axhline(0, color='k', lw=0.5, alpha=0.5)
# negative lyap exp
ax2.plot(r[lyap < 0],
         lyap[lyap < 0] / iterations,
         '.k', alpha=0.5, ms=0.5)
# positive lyap exp
ax2.plot(r[lyap >= 0],
         lyap[lyap >= 0] / iterations,
         '.r', alpha=0.5, ms=0.5)
ax2.set_xlim(2.5, 4)
ax2.set_ylim(-2, 1)
ax2.set_title("Lyapunov exponent")
plt.tight_layout()

plt.show()
