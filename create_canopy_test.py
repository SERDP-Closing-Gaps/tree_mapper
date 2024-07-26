import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import rioxarray as rio
from scipy.ndimage import gaussian_filter
from numba import jit
import numpy.random as random
from scipy.interpolate import NearestNDInterpolator
from tree_sampling import sample_trees, get_synthetic_chm
from scipy.special import beta

# CHM Data
chm_data = rio.open_rasterio('data/chm.tif')
chm = chm_data.data[0]
xx, yy = np.meshgrid(chm_data.x, chm_data.y)

tree_inventory = gpd.read_file('data/tree_inventory.geojson')
print(tree_inventory)
quit()

x = tree_inventory['X'].to_numpy()
y = tree_inventory['Y'].to_numpy()
coords = np.c_[x, y]


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