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
    theta = i/(nlines*2*math.pi)
    intersections = np.column_stack(intersections, )