import numpy as np
from numba import jit
import numpy.random as random
from skimage.feature import peak_local_max
import math 
from scipy.special import beta

"""
Randomly generates tree locations given a crown height map and a radius 
field that defines the minimum spacing between trees. 
"""

@jit
def acceptance_ratio(dist : float, height0 : float, height1 : float, radius : float):
     
    np.expit((dist - radius))


def sample_trees(chm_data, radius):

    # Tree mask
    mask = np.zeros_like(chm_data)
    mask[chm_data > 1.] = 1.

    # Get seed points distributed over different tree clusters
    active = list(map(tuple, peak_local_max(chm_data, min_distance=5, threshold_abs = 1.)))
    component_points = np.array(active)

    # Pixel size (m)
    dx = 1.

    # Mark the seed points
    n0 = mask.shape[0]
    n1 = mask.shape[1]
    marked = np.zeros((n0, n1), dtype=np.int64)
    marked[component_points[:,0], component_points[:,1]] = 1

    # Coordinate grid
    xs = np.arange(0., n0, 1.)
    ys = np.arange(0., n1, 1.)
    xx, yy = np.meshgrid(ys, xs)

    @jit(nopython=True)
    def disk_sample(xx, yy, marked, mask, active, radius, dx=1., K=25):

        while len(active) > 0:
            # Randomly select an active point
            index = random.randint(0, len(active))
            i, j = active[index]
            found = False

            for k in range(K):
                # Randomly generate a point in vicinity of active point
                i1, j1 = get_point(i, j, radius, dx)
                valid = is_valid(xx, yy, marked, mask, i1, j1, radius, dx) 

                if valid:
                    marked[i1, j1] = 1
                    active.append((i1, j1))
                    found = True
                    break

            if not found:
                i, j = active.pop(index)

        return marked


    @jit(nopython=True)
    def get_point(i, j, radius, dx):
        # Randomly generate a new point nearby point i,j
        px = i*dx 
        py = j*dx
        r = 2.*dx + 2.*radius[i,j]*random.random()
        theta =  2. * np.pi * random.random()
        px1 = px + r*np.cos(theta)
        py1 = py + r*np.sin(theta)
        i1 = int(px1 / dx)
        j1 = int(py1 / dx)

        return (i1, j1)


    @jit(nopython=True)
    def is_valid(xx, yy, marked, chm, i, j, radius, dx):

        # Check to make sure point is in bounds 
        n0 = marked.shape[0]
        n1 = marked.shape[1]
        valid = (1 <= i and i <= n0-1 and 1 <= j and j <= n1-1 and chm[i,j] > 0.)

        if not valid:
            return valid
    
        # Check a local window around point and see if any points are too close to proposed sample
        r = radius[i,j]
        w = int(r / dx) + 1
        sub_xx = xx[max(0,i-w):min(n0, i+w), max(0,j-w):min(n1, j+w)]
        sub_yy = yy[max(0,i-w):min(n0, i+w), max(0,j-w):min(n1, j+w)]
        sub_marked = marked[max(0,i-w):min(n0, i+w), max(0,j-w):min(n1, j+w)]

        px = xx[i,j]
        py = yy[i,j]

        d = np.sqrt((sub_xx - px)**2 + (sub_yy - py)**2)
        
        min_dist = 2.*r
        for i1 in range(sub_marked.shape[0]):
            for j1 in range(sub_marked.shape[1]):
                if sub_marked[i1,j1] == 1:
                    min_dist = min(min_dist, d[i1,j1])
                    if min_dist < r:
                        return False

        return True


    marked = disk_sample(xx, yy, marked, mask, active, radius, dx=dx)
    return marked
 

"""
Generates a synthetic canopy height model from a list of trees.
"""

def get_synthetic_chm(chm, tree_indexes, tree_params, dx=1.):

    xs = np.arange(-5., 5.+dx, dx)
    ys = np.arange(-5., 5.+dx, dx)
    xx, yy = np.meshgrid(xs, ys)
    r = np.sqrt(xx**2 + yy**2)
    beta0 = beta(tree_params[:,0], tree_params[:,1])

    chm = _get_synthetic_chm(chm, r, tree_indexes, tree_params, beta0, dx)
    return chm

@jit(nopython=True)
def _get_synthetic_chm(chm, r, tree_indexes, tree_params, beta0, dx=1.):

        n = r.shape[0]
        n0 = math.floor(n/2)
        

        for k in range(len(tree_indexes)):
            i = tree_indexes[k][0]
            j = tree_indexes[k][1]

            # Tree and beta distribution params
            a = tree_params[k][0]
            b = tree_params[k][1]
            c = tree_params[k][2]
            height = tree_params[k][3]
            crown_length = tree_params[k][4]

            tree_map = get_tree_height(r, a, b, c, height, crown_length,  beta0[k])

            for i1 in range(r.shape[0]):
                for j1 in range(r.shape[1]):
                    chm[i1 + i - n0, j1 + j - n0] = max(tree_map[i1, j1], chm[i1 + i - n0, j1 + j - n0])
            
        return chm

@jit(nopython=True)
def get_tree_height(r, a, b, c, height, crown_length, beta0, dx=1.):

    r = r.copy()

    # z position of max radius
    z_max =  (a - 1.) / (a + b - 2.)
    # max tree radius
    r_max = crown_length * c * z_max**(a - 1.) * (1. - z_max)**(b - 1)

    # Initial guess for newton iteration
    z0 = np.ones_like(r)*0.7

    C = crown_length * c / beta0

    # Newton iteration to find tree height at given radius
    for k in range(5):
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                r0 =  C * z0[i,j]**(a - 1.) * (1. - z0[i,j])**(b - 1) - min(r[i,j],r_max)
                dr_dz = C * z0[i,j]**(a - 1) * (-(1.-z0[i,j])**(b-2))*(a*(z0[i,j]-1.) 
                                                + (b-2.)*z0[i,j] + 1.)
                z0[i,j] = max(z0[i,j], z_max)
                z0[i,j] = min(z0[i,j], 1.)
                z0[i,j] = z0[i,j] - 0.25*(r0 / dr_dz) 

    z0 = z0*crown_length + (height-crown_length)

    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            if r[i,j] > r_max:
                z0[i,j] = 0.

    return z0



