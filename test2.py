import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from numba import vectorize, float64, jit, njit

@vectorize([float64(float64, float64, float64, float64, float64)])
def get_radius(z, crown_base, crown_height, dbh, trait_score):
    height = crown_base + crown_height

    if z < crown_base:
        return 0.0
    
    if z > height:
        return 0.
    
    C0_R0 = 0.503
    C1_R0 = 3.126
    C0_R40 = 0.5
    C1_R40 = 10.0
    C0_B = 0.196
    C1_B = 0.511

    r0j = (1 - trait_score) * C0_R0 + trait_score * C1_R0
    r40j = (1 - trait_score) * C0_R40 + trait_score * C1_R40
    max_crown_radius = r0j + (r40j - r0j) * (dbh / 40.0)
    shape_parameter = (1 - trait_score) * C0_B + trait_score * C1_B

    return max_crown_radius * ((height - z) / height)**shape_parameter


@njit(nopython=True)
def intersects(distance, props0, props1):

    crown_base0, crown_length0, dbh0, trait_score0 = props0 
    crown_base1, crown_length1, dbh1, trait_score1 = props1

    cb_max = max(crown_base0, crown_base1)

    r0_max = get_radius(cb_max, crown_base0, crown_length0, dbh0, trait_score0)
    r1_max = get_radius(cb_max, crown_base1, crown_length1, dbh1, trait_score1)

    return r0_max + r1_max >= distance


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
    crown_ratio = 0.4*random.random() + 0.5
    crown_base = height - crown_ratio*height
    crown_length = height - crown_base
    dbh = 11. 
    trait_score = 0.53

    return (crown_base, crown_length, dbh, trait_score)


@njit(nopython=True)
def propose_understory_tree(chm, i, j):
    height = 20. + random.random()*30.
    crown_ratio = 0.3*random.random() + 0.6
    crown_base = height - crown_ratio*height
    crown_length = height - crown_base
    dbh = 11. 
    trait_score = 0.53

    return (crown_base, crown_length, dbh, trait_score)

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
    w = int(3*r / dx) + 1
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
            di, dj = propose_point(radius[i,j], 5*radius[i,j])
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


n = 4001
# crown base, crown length, dbh, species
tree_props = [(5., 10., 11., 0.)]
tree_grid = np.zeros((n, n), dtype=np.int32)
chm = 10.*np.ones((n, n), dtype=np.float32)
radius = 2.*np.ones((n, n))
candidate_trees = [(2000, 2000)]
tree_grid[2000,2000] = 1

tree_grid, tree_props = disk_sample(tree_grid, chm, radius, candidate_trees, tree_props, propose_tree = propose_understory_tree)

tree_props = np.array(tree_props)
height = tree_props[:,0] + tree_props[:,1]

heights = np.zeros_like(tree_grid)

indexes = np.argwhere(tree_grid > 0)
prop_indexes = tree_grid[indexes[:,0], indexes[:,1]] - 1
heights[indexes[:,0], indexes[:,1]] = height[prop_indexes]

plt.imshow(heights)
plt.colorbar()
plt.show()



#heights[tree_grid > 0] =  height[t]

#print(height)
quit()

indexes = np.argwhere(tree_grid > 0)
candidate_trees = list(zip(indexes[:,0], indexes[:,1]))
radius = 2.*np.ones((n, n))


tree_grid, tree_props = disk_sample(tree_grid, chm, radius, candidate_trees, tree_props, propose_tree = propose_understory_tree)

#print(len(candidate_trees))
#print(len(tree_props))

plt.imshow(tree_grid)
plt.show()


