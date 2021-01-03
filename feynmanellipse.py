"""
Zachary Weiss
2 Jan 2021
Ellipse construction as described by 3B1B in his video on Feynman's Lost Lecture
"""
import numpy as np
import matplotlib.pyplot as plt
import math

nlines = 50
radius = 1
point = [0.9, 0]

if sum(np.square(point)) >= radius:
    raise ValueError('Point is outside of or on circle')

intersections = []
for i in range(0, nlines-1):
    # can likely do this smarter array-wise, but just implementing fast here
    theta = (i/nlines)*2*math.pi
    b = 2*(point[0]*np.cos(theta)+point[1]*np.sin(theta))
    intersections = np.hstack(intersections, [x])