"""
Zachary Weiss
2 Jan 2021
Ellipse construction as described by 3B1B in his video on Feynman's Lost Lecture
NOTE: Purposed modified partway through, as intermediate result more interesting than planned ellipse
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import math

n_lines = 500
radius = 1  # not much reason to change from 1 as only serves to scale (relative to other params)
point = [0.5, 0]

if sum(np.square(point)) >= radius:
    raise ValueError('Point is on or outside the defined circle')

fig = plt.figure()
ax = fig.add_subplot(111)

theta = np.zeros((n_lines,), dtype=float)
dist = np.zeros((n_lines,), dtype=float)
intersections = np.zeros((n_lines, 2), dtype=float)
midpts = np.zeros((n_lines, 2), dtype=float)
tangents = np.zeros((n_lines, 2, 2), dtype=float)
for i in range(0, n_lines):
    # can likely do this smarter array-wise, but just quickly implementing as a loop
    theta[i] = (i / n_lines) * 2 * math.pi
    components = np.array([np.cos(theta[i]), np.sin(theta[i])])
    b = 2*np.dot(point, components)
    c = sum(np.square(point))-np.square(radius)

    dist[i] = 0.5*(-b+np.sqrt(np.square(b)-4*c))
    vec = np.array(dist[i]*components)
    intersections[i, :] = np.add(vec, point)
    midpts[i, :] = np.add(vec/2, point)

    tan_vec = np.array(dist[i] * np.flip(components)) / 2
    # the transpose was a mistake, but it generated a fantastic end result so I'm pausing here to play around w/ it
    tangents[i, :, :] = np.array([np.add(tan_vec, midpts[i, :]), np.add(-tan_vec, midpts[i, :])]).transpose()
    #tangents[i, :, :] = np.array([np.add(tan_vec, midpts[i, :]), np.add(-tan_vec, midpts[i, :])])


lc = mc.LineCollection(tangents)
ax.add_collection(lc)
ax.autoscale()

plt.grid(False)
plt.axis('off')
plt.tight_layout()
ax.set_aspect('equal')
plt.show()
