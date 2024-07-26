import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from numba import vectorize, float64, jit, njit
import math
import plotly.graph_objects as go

a = 1.1821
b = 1.4627
c = 0.1528

@vectorize([float64(float64, float64, float64)])
def get_radius(z,  crown_base, crown_length):

    height = crown_base + crown_length

    z = (z-crown_base) / height

    if z <= 0.:
        return 0.
    if z >= 1.:
        return 0.

    r =  2.5*crown_length * c * z**(a - 1.) * (1. - z)**(b - 1)
    return r


@njit(nopython=True)
def intersects(distance, props0, props1):
    crown_base0, crown_length0 = props0 
    crown_base1, crown_length1 = props1

    height0 = crown_base0 + crown_length0
    height1 = crown_base1 + crown_length1 
    overlap_min = max(crown_base0, crown_base1)
    overlap_max = min(height0, height1)

    if overlap_min < overlap_max:
        z = overlap_min 
        
        while z <= overlap_max:
            r0 = get_radius(z, crown_base0, crown_length0)
            r1 = get_radius(z, crown_base1, crown_length1)    
            if r0 + r1 > distance:
                return True 
            
            z += max(min(1., overlap_max-z), 1e-3)
    
    return False


@njit(nopython=True)
def propose_point(min_radius, max_radius, dx=1.):

    r = min_radius + random.random()*(max_radius - min_radius)
    theta =  2. * np.pi * random.random()
    di = r*np.cos(theta)
    dj = r*np.sin(theta)
    di = int(di / dx)
    dj = int(dj / dx)

    return (di, dj)

@njit(nopython=True)
def propose_overstory_tree(chm, i, j):

    height = chm[i,j]
    crown_ratio = 0.6*random.random() + 0.3
    crown_base = height - crown_ratio*height
    crown_length = height - crown_base

    return (crown_base, crown_length)


@njit(nopython=True)
def propose_understory_tree(chm, i, j):
    height = 1. + random.random()*20.
    crown_ratio = 0.3*random.random() + 0.6
    crown_base = height - crown_ratio*height
    crown_length = height - crown_base

    return (crown_base, crown_length)

@njit
def is_valid(chm, tree_grid, tree_props, proposed_tree_props, i, j, radius, dx):
    # Check to make sure point is in bounds 
    n0 = tree_grid.shape[0]
    n1 = tree_grid.shape[1]
    valid = (1 <= i and i <= n0-1 and 1 <= j and j <= n1-1 and chm[i,j] > 1.)

    if not valid:
        return valid

    # Check a local window around point and see if any points are too close to proposed sample
    r = radius[i, j]
    w = int(4*r / dx) + 1
    for i1 in range(max(0,i-w), min(n0, i+w)):
        for j1 in range(max(0,j-w), min(n1, j+w)):
            if tree_grid[i1,j1] > 0:
                # Get tree properties
                props = tree_props[tree_grid[i1, j1] - 1]
                
                dist = np.sqrt(((i - i1)*dx)**2 +  ((j - j1)*dx)**2)
                if dist < r:
                    return False
                
                collision = intersects(dist, props, proposed_tree_props)
                if collision:
                    return False
                
    return True



@njit
def disk_sample(tree_grid, chm, radius, candidate_trees, tree_props, dx=1., K=25, propose_tree = propose_overstory_tree):
    
    while len(candidate_trees) > 0:

        index = random.randint(0, len(candidate_trees))
        i, j = candidate_trees[index]
        found = False

        for k in range(K):
            di, dj = propose_point(radius[i,j], 4*radius[i,j])
            i1 = i + di 
            j1 = j + dj
            proposed_tree_props = propose_tree(chm, i1, j1)
            valid = is_valid(chm, tree_grid, tree_props, proposed_tree_props, i1, j1, radius, dx)

            if valid:
                candidate_trees.append((i1, j1))
                tree_props.append(proposed_tree_props)
                tree_grid[i1, j1] = len(tree_props)
                found = True
                break 

        if not found:
            i, j = candidate_trees.pop(index)

    return tree_grid, tree_props


n = 41
# crown base, crown length, dbh, species
tree_props = [(5., 15.)]
tree_grid = np.zeros((n, n), dtype=np.int32)
chm = 10.*np.ones((n, n), dtype=np.float32)
radius = 2.*np.ones((n, n))
candidate_trees = [(20, 20)]
tree_grid[10, 10] = 1

tree_grid, tree_props = disk_sample(tree_grid, chm, radius, candidate_trees, tree_props, propose_tree = propose_understory_tree)

tree_props = np.array(tree_props)
height = tree_props[:,0] + tree_props[:,1]

heights = np.zeros_like(tree_grid)

indexes = np.argwhere(tree_grid > 0)
prop_indexes = tree_grid[indexes[:,0], indexes[:,1]] - 1
#heights[indexes[:,0], indexes[:,1]] = height[prop_indexes]

tree_props = tree_props[prop_indexes]


### PLOT


z_points = 31
# Define the number of angular points for the radial symmetry
theta_points = 40
# Generate the z values
z = np.linspace(0, 30., z_points)
# Generate the theta values
theta = np.linspace(0, 2 * np.pi, theta_points)
# Create a meshgrid for theta and z
theta, z = np.meshgrid(theta, z)

# Create the plot
fig = go.Figure()


for j in range(len(indexes)):

    props = tree_props[j]

    crown_base = props[0]
    crown_length = props[1]

    radius = get_radius(z, crown_base, crown_length)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    fig.add_trace(go.Surface(x=x + float(indexes[j,0]), y=y+float(indexes[j,1]), z=z, opacity=0.5))



fig.update_layout(
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
        aspectmode='cube'  # Set the aspect ratio mode to 'cube'
    ),
    title='Radially Symmetric Objects'
)
fig.show()
