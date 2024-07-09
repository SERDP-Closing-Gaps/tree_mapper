import numpy as np
from numba import jit, types
import random
import matplotlib.pyplot as plt
import math
from numba.typed import List

dx = 1.
N = 30
sample_grid = np.zeros((N,N))
radius = 5.*np.ones((N,N))

@jit(nopython=True)
def fits(sample_grid, radius, i, j, dx):


    # Get a window around the current point

    #R = radius[i,j]
    return False
   
    
@jit(nopython=True)
def in_bounds(sample_grid, i, j):
     return (0 <= i and i <= sample_grid.shape[0] and 0 <= j and j <= sample_grid.shape[1])

@jit(nopython=True)
def f(sample_grid, r, pi, pj, K=30, dx=1.):
    active = []
    active.append([pi, pj])
    sample_grid[pi, pj] = 1.
    z = 0

    while z < 100:
     
        # Randomly select an active point
        index = random.randint(0, len(active)-1)

        pi, pj = active[index]
        px = dx*pi
        py = dx*pj
        found = False

        # Get the characteristic radius at point
        R = radius[pi, pj]

        for k in range(K):
            # Randomly generate a new point nearby
            r = R*(random.random() + 1.)
            theta =  2. * np.pi * random.random()
            px1 = px + r*np.cos(theta)
            py1 = py + r*np.sin(theta)

            i1 = int(px1 / dx)
            j1 = int(py1 / dx)

            if in_bounds(sample_grid, i1, j1) and fits(sample_grid, radius, i1, j1, dx):
                sample_grid[i1, j1] = 1.
                active.append([i1, j1])
                found = True 
                break

        z += 1

        if not found:
            coords = active.pop(index)
            t1 = coords[0]
            t2 = coords[1]
            sample_grid[t1,t2] = 0.


@jit(nopython=False)
def disk_sample(sample_grid, i0, j0, r=5., dx=1.):
    # List of tuples of array indexes of active nodes
    active = List()
    
    # Add tuples to the list
    for i in range(10):
        lst.append((i, i + 1))
    
    return lst

l = tuple_list_example()
print(l)
quit()


@jit(nopython=False)
def tuple_list_example():
    # Create an empty Numba list with a tuple of two integers as its type
    lst = List.empty_list(types.Tuple((types.int32, types.int32)))
    
    # Add tuples to the list
    for i in range(10):
        lst.append((i, i + 1))
    
    return lst

l = tuple_list_example()
print(l)
quit()

@jit(nopython=False)
def tuple_list_example():
    # Create an empty Numba list with a tuple of two integers as its type
    lst = List.empty_list(types.Tuple((types.int32, types.int32)))
    
    # Add tuples to the list
    for i in range(10):
        lst.append((i, i + 1))
    
    return lst

l = tuple_list_example()
print(l)
quit()

f(sample_grid, radius, 15, 15)
plt.imshow(sample_grid)
plt.colorbar()
plt.show()