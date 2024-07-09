import numpy as np
from numba import jit, types
import random
import matplotlib.pyplot as plt
import math
from numba.typed import List

dx = 1.
n0 = 500
n1 = 500
marked = np.zeros((n0, n1), dtype=np.int64)
dx = 1.
xs = np.arange(0., n0, 1.)
ys = np.arange(0., n1, 1.)
xx, yy = np.meshgrid(xs, ys)
#radius = 5.*np.ones((n0, n1))


@jit(nopython=True)
def disk_sample(xx, yy, marked, i0, j0, radius=10., dx=1., K=30):
    # List of tuples of array indexes of active nodes
    active = List()
    active.append((i0,j0))
    marked[i0, j0] = 1 

    z = 0

    while len(active) > 0:
        # Randomly select an active point
        index = random.randint(0, len(active)-1)
        i, j = active[index]
        found = False

        for k in range(K):
            # Randomly generate a point in vicinity of active point
            i1, j1 = get_point(i, j, radius, dx)

            valid = is_valid(xx, yy, marked, i1, j1, radius, dx) 

            if valid:
                marked[i1, j1] = 1
                active.append((i1, j1))
                found = True
                break
                
        z += 1

        if not found:
            i, j = active.pop(index)
            #marked[i, j] = 0

    return marked, active


@jit(nopython=True)
def get_point(i, j, radius, dx):
    # Randomly generate a new point nearby point i,j
    px = i*dx 
    py = j*dx
    r = radius*(random.random() + 1.)
    theta =  2. * np.pi * random.random()
    px1 = px + r*np.cos(theta)
    py1 = py + r*np.sin(theta)
    i1 = int(px1 / dx)
    j1 = int(py1 / dx)

    return (i1, j1)



@jit(nopython=True)
def is_valid(xx, yy, marked, i, j, radius, dx):

    # Check to make sure point is in bounds 
    n0 = marked.shape[0]
    n1 = marked.shape[1]
    valid = (1 <= i and i <= n0-1 and 1 <= j and j <= n1-1)

    if not valid:
        return valid
   
    # Check a local window around point and see if any points are too close
    w = int(radius / dx) + 1
    sub_xx = xx[max(0,i-w):min(n0, i+w), max(0,j-w):min(n1, j+w)]
    sub_yy = yy[max(0,i-w):min(n0, i+w), max(0,j-w):min(n1, j+w)]
    sub_marked = marked[max(0,i-w):min(n0, i+w), max(0,j-w):min(n1, j+w)]

    px = xx[i,j]
    py = yy[i,j]

    d = np.sqrt((sub_xx - px)**2 + (sub_yy - py)**2)
    

    min_dist = 2.*radius
    for i1 in range(sub_marked.shape[0]):
        for j1 in range(sub_marked.shape[1]):
            if sub_marked[i1,j1] == 1:
                min_dist = min(min_dist, d[i1,j1])
                if min_dist < radius:
                    return False

    return True


marked, active = disk_sample(xx, yy, marked, 250, 250, 10.)

plt.imshow(marked)
plt.show()
