import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import rioxarray as rio
from numba import vectorize, float64, jit, njit
from scipy.interpolate import NearestNDInterpolator
from skimage.feature import peak_local_max
import numpy.random as random
import json


with open('data/spcd_parameters.json') as f:
    spcd_params = json.load(f)


# CHM Data
chm_data = rio.open_rasterio('data/chm.tif')
chm = chm_data.data[0]
xx, yy = np.meshgrid(chm_data.x, chm_data.y)

# Tree inventory
tree_inventory = gpd.read_file('data/tree_inventory.geojson')
tree_inventory = tree_inventory.sample(frac=1)
x = tree_inventory['X'].to_numpy()
y = tree_inventory['Y'].to_numpy()
tree_coords = np.c_[x, y]

# Tree properties as numpy array
tree_props = np.column_stack([
    x,
    y, 
    tree_inventory['DIA'].to_numpy(),
    tree_inventory['HT'].to_numpy(),
    tree_inventory['CR'].to_numpy(),
    tree_inventory['TRAIT_SCORE'].to_numpy(),
])

# Cluster coordinates
cluster_indexes = peak_local_max(chm, min_distance=5, threshold_abs = 1.)
cluster_xs = xx[cluster_indexes[:,0], cluster_indexes[:,1]]
cluster_ys = yy[cluster_indexes[:,0], cluster_indexes[:,1]]
cluster_coords = np.c_[cluster_xs, cluster_ys]

# Tree distances to clusters
nbrs = NearestNeighbors(n_neighbors=40, algorithm='ball_tree').fit(tree_coords)
distances, clusters = nbrs.kneighbors(cluster_coords)

# Voroni map of clusters
cluster_interp = NearestNDInterpolator(cluster_coords, np.arange(len(cluster_coords)))
cluster_map = cluster_interp(xx, yy).astype(int)

"""
Propose a point in a given radius. 
"""
@njit
def propose_point(min_radius : float, max_radius : float, dx=1.):

    r = min_radius + random.random()*(max_radius - min_radius)
    theta =  2. * np.pi * random.random()
    di = r*np.cos(theta)
    dj = r*np.sin(theta)
    di = int(di / dx)
    dj = int(dj / dx)

    return (di, dj)

@njit 
def initialize_trees(tree_grid, clusters, cluster_indexes, tree_props):

    active_trees = []

    # Find the largest tree in each cluster
    for i in range(len(cluster_indexes)):
        cluster = clusters[i]
        x = cluster_indexes[i,0]
        y = cluster_indexes[i,1]

        max_height = 0 
        tallest_tree_index = 0
        for j in range(len(cluster)):
            tree_index = cluster[j]
            props = tree_props[tree_index]
            height = props[4]

            if height > max_height:
                max_height = height 
                tallest_tree_index = tree_index

        tree_grid[x, y] = tallest_tree_index + 1
        active_trees.append((x,y))

    return tree_grid, active_trees


tree_grid = np.zeros_like(chm, dtype=np.int64)
tree_grid, active_trees = initialize_trees(tree_grid, clusters, cluster_indexes, tree_props)

@vectorize([float64(float64, float64, float64, float64, float64)])
def get_radius(z, height, crown_base, dbh, trait_score):

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


@njit
def intersects(distance, props0, props1, scale=1.):
    
    dbh0 = props0[2]
    height0 = props0[3]
    crown_ratio0 = props0[4]
    trait_score0 = props0[5]
    crown_base0 = height0 - crown_ratio0*height0

    dbh1 = props1[2]
    height1 = props1[3]
    crown_ratio1 = props1[4]
    trait_score1 = props1[5]
    crown_base1 = height1 - crown_ratio1*height1

    cb_max = max(crown_base0, crown_base1)
    r0_max = scale*get_radius(cb_max, height0, crown_base0, dbh0, trait_score0)
    r1_max = scale*get_radius(cb_max, height1, crown_base1, dbh1, trait_score1)

    return r0_max + r1_max >= distance



@njit
def is_valid(chm, tree_grid, tree_props, tree_index, i, j, min_radius=2., radius=10., dx=1.):
    # Check to make sure point is in bounds 
    n0 = tree_grid.shape[0]
    n1 = tree_grid.shape[1]
    valid = (1 <= i and i <= n0-1 and 1 <= j and j <= n1-1 and chm[i,j] > 1.)
    proposed_tree_props = tree_props[tree_index]

    if not valid:
        return valid

    # Check a local window around point and see if any points are too close to proposed sample
    w = int(radius / dx) + 1
    for i1 in range(max(0,i-w), min(n0, i+w)):
        for j1 in range(max(0,j-w), min(n1, j+w)):
            if tree_grid[i1,j1] > 0:
                # Get tree properties
                props = tree_props[tree_grid[i1, j1] - 1]
                
                dist = np.sqrt(((i - i1)*dx)**2 +  ((j - j1)*dx)**2)
                if dist < min_radius:
                    return False
                
                collision = intersects(dist, props, proposed_tree_props)
                if collision:
                    return False
                
    return True


@njit
def disk_sample(tree_grid, chm, clusters, cluster_map, candidate_trees, tree_props, dx=1., K=25):
    while len(candidate_trees) > 0:

        index = random.randint(0, len(candidate_trees))
        i, j = candidate_trees[index]
        found = False
        #z += 1

        for k in range(K):
            di, dj = propose_point(2.,10.)
            i1 = i + di 
            j1 = j + dj

            # Get the cluster to which this tree belongs 
            cluster_index = cluster_map[i1, j1]
            # Then choose a random tree near this cluster
            n = random.randint(0, len(clusters[cluster_index]))
            tree_index = clusters[cluster_index][n]
            
            valid = is_valid(chm, tree_grid, tree_props, tree_index, i1, j1)
            #print(k, i, j, i1, j1, valid)

            if valid:
                candidate_trees.append((i1, j1))
                tree_grid[i1, j1] = tree_index
                found = True
                break 

        if not found:
            i, j = candidate_trees.pop(index)

    return tree_grid

tree_grid = disk_sample(tree_grid, chm, clusters, cluster_map, active_trees, tree_props)
tree_grid[tree_grid > 1] = 1

plt.subplot(2,1,1)
plt.imshow(tree_grid)

plt.subplot(2,1,2)
plt.imshow(chm)

plt.show()
quit()


indexes = tree_grid[tree_grid >= 1]

vals, counts = np.unique(indexes, return_counts=True)
print(vals)
print(counts)
quit()


indexes = np.sort(indexes)
print(indexes)
quit()

plt.imshow(tree_grid)
plt.colorbar()
plt.show()

quit()


plt.imshow(chm)
plt.scatter(clusters[:,1], clusters[:,0], color="red", s=0.5, alpha=0.5)
plt.show()

quit()



# Generate a radius field that controls tree spacing
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(coords)
distances, indices = nbrs.kneighbors(coords)
distances = distances[:,1:]

radius = distances.min(axis=1)
radius_interp = NearestNDInterpolator(coords, radius)
radius = radius_interp(xx, yy)
radius = gaussian_filter(radius, sigma=5.)

# Sample tree locations
marked = sample_trees(chm, radius)


# From the tree, locations generate a chm
tree_indexes = np.argwhere(marked > 0.)
n = len(tree_indexes)
a = np.ones(n) * 1.1821
b = np.ones(n) * 1.4627
c = np.ones(n) * 0.1528
height = chm[tree_indexes[:,0], tree_indexes[:,1]]#*2.5
crown_length = height
pad = 20
half_pad = int(pad/2)
tree_indexes += half_pad
synthetic_chm = np.zeros((chm.shape[0] + pad, chm.shape[1] + pad))
tree_params = np.column_stack([a, b, c, height, crown_length])



r = get_synthetic_chm(synthetic_chm, tree_indexes, tree_params)
r = r[half_pad:-half_pad, half_pad:-half_pad]


tree_indexes -= half_pad
print(tree_indexes.shape)

plt.subplot(3,1,1)
plt.title('Synthetic Canopy Height Model')
plt.imshow(r)
plt.colorbar()


plt.subplot(3,1,2)
plt.title('Tree Locations')
plt.imshow(chm)
plt.colorbar()

plt.scatter(tree_indexes[:,1], tree_indexes[:,0], color="red", s=0.5, alpha=0.5)


plt.subplot(3,1,3)
plt.title('Meta Canopy Height Model')
plt.imshow(chm)
plt.colorbar()
plt.show()