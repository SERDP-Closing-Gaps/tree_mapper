import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import rioxarray as rio
from numba import jit, njit 
from scipy.interpolate import NearestNDInterpolator
from skimage.feature import peak_local_max
from numba.typed import List
import numpy.random as random

# CHM Data
chm_data = rio.open_rasterio('data/chm.tif')
chm = chm_data.data[0]
xx, yy = np.meshgrid(chm_data.x, chm_data.y)

# Tree inventory
tree_inventory = gpd.read_file('data/tree_inventory.geojson')
tree_inventory = tree_inventory.sample(frac=1)
x = tree_inventory['X'].to_numpy()
y = tree_inventory['Y'].to_numpy()
coords = np.c_[x, y]

# Tree properties as numpy array
tree_props = np.column_stack([
    x,
    y, 
    tree_inventory['SPCD'].to_numpy(),
    tree_inventory['DIA'].to_numpy(),
    tree_inventory['HT'].to_numpy(),
    tree_inventory['CR'].to_numpy(),
])

# Cluster coordinates
clusters = peak_local_max(chm, min_distance=5, threshold_abs = 1.)
cluster_xs = xx[clusters[:,0], clusters[:,1]]
cluster_ys = yy[clusters[:,0], clusters[:,1]]
cluster_coords = np.c_[cluster_xs, cluster_ys]

# Tree distances to clusters
nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(cluster_coords)
distances, nearby_groves = nbrs.kneighbors(coords)

# List of groves
groves = List(List((clusters[i,0], clusters[i,1])) for i in range(len(clusters)))  

active_groves = np.ones(len(groves), dtype=bool)
tree_grid = np.zeros_like(chm)

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



"""
Attempt to add a tree to a given grove.
"""
@njit
def add_tree_to_grove(tree_index, tree_props, grove):

    if len(grove) == 1:        
        # Place tree here
        x, y = grove[0]
        tree_grid[x, y] = tree_index
    else:
        # Randomly select an active tree in grove
        n = random.randint(0, len(grove)-1)
        x, y  = grove[n]

        # Try to fit a tree near it
        dx, dy = propose_point(2., 10.)
        x1 = x + dx 
        y1 = y + dx 


@njit 
def distribute_trees(tree_props, chm, nearby_groves, groves, active_groves):

    for i in range(len(tree_props)):

        # Look at what groves are near this tree
        for j in range(len(nearby_groves[i])):
            grove_index = nearby_groves[i,j]                       

            # See if we can add a tree to this grove
            if active_groves[grove_index]:
                grove = groves[grove_index]
                success = add_tree_to_grove(i, tree_props, grove)





                print(grove)

distribute_trees(tree_props, chm, nearby_groves, groves, active_groves)
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