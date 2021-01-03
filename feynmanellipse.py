"""
Zachary Weiss
2 Jan 2021
Ellipse construction as described by 3B1B in his video on Feynman's Lost Lecture
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import math


nlines = 500
radius = 1
point = [0.5, 0]

if sum(np.square(point)) >= radius:
    raise ValueError('Point is on or outside the defined circle')

fig = plt.figure()
ax = fig.add_subplot(111)

theta = np.zeros((nlines,), dtype=float)
dist = np.zeros((nlines,), dtype=float)
intersections = np.zeros((nlines, 2), dtype=float)
midpts = np.zeros((nlines, 2), dtype=float)
tangents = np.zeros((nlines, 2, 2), dtype=float)
for i in range(0, nlines):
    # can likely do this smarter array-wise, but just quickly implementing as a loop
    theta[i] = (i/nlines)*2*math.pi
    components = np.array([np.cos(theta[i]), np.sin(theta[i])])
    b = 2*np.dot(point, components)
    c = sum(np.square(point))-np.square(radius)

    dist[i] = 0.5*(-b+np.sqrt(np.square(b)-4*c))
    vec = np.array(dist[i]*components)
    intersections[i, :] = np.add(vec, point)
    midpts[i, :] = np.add(vec/2, point)

    tanvec = np.array(dist[i]*np.flip(components))/2
    # the transpose was a mistake, but it generated a fantastic end result so I'm pausing here to play around w/ it
    tangents[i, :, :] = np.array([np.add(tanvec, midpts[i, :]), np.add(-tanvec, midpts[i, :])]).transpose()


lc = mc.LineCollection(tangents)
ax.add_collection(lc)
ax.autoscale()

plt.grid(False)
plt.axis('off')
ax.set_aspect('equal')
plt.show()
