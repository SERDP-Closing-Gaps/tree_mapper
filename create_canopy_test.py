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
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
import math
from scipy.special import beta

# CHM Data
chm_data = rio.open_rasterio('data/chm.tif')
chm = chm_data.data[0]
xx, yy = np.meshgrid(chm_data.x, chm_data.y)

plt.imshow(chm)
plt.show()
quit()

# Tree inventory
data = pd.read_csv('data/tree_inventory.csv')
tree_inventory = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data["X"], data["Y"]), crs="EPSG:4326")
tree_inventory = tree_inventory.to_crs(3857)
tree_inventory["X"] = tree_inventory.geometry.x
tree_inventory["Y"] = tree_inventory.geometry.y
tree_inventory = tree_inventory.dropna()
x = tree_inventory['X'].to_numpy()
y = tree_inventory['Y'].to_numpy()
coords = np.c_[x, y]

nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(coords)
distances, indices = nbrs.kneighbors(coords)
distances = distances[:,1:]

radius = distances.min(axis=1)
radius_interp = NearestNDInterpolator(coords, radius)
radius = radius_interp(xx, yy)
radius = gaussian_filter(radius, sigma=5.)


tree_indexes = peak_local_max(chm, min_distance=5, threshold_abs = 1.)
pad = 20
tree_indexes += int(pad/2)
synthetic_chm = np.zeros((chm.shape[0] + pad, chm.shape[1] + pad))
#synthetic_chm[tree_indexes[:,0], tree_indexes[:,1]] = 1.

n = len(tree_indexes)
a = np.ones(n) * 1.1821
b = np.ones(n) * 1.4627
c = np.ones(n) * 0.1528
height = np.ones(n) * 30.
crown_length = np.ones(n) * 30.
beta0 = beta(a, b)
tree_params = np.column_stack([a, b, c, height, crown_length, beta0])

def get_synthetic_chm(chm, tree_indexes, tree_params, dx=1.):

    xs = np.arange(-5., 5.+dx, dx)
    ys = np.arange(-5., 5.+dx, dx)
    xx, yy = np.meshgrid(xs, ys)
    r = np.sqrt(xx**2 + yy**2)

    chm = _get_synthetic_chm(chm, r, tree_indexes, tree_params, dx)
    return chm

@jit(nopython=True)
def _get_synthetic_chm(chm, r, tree_indexes, tree_params, dx=1.):

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
            beta0 = tree_params[k][5]

            tree_map = get_tree_height(r, a, b, c, height, crown_length, beta0)

            for i1 in range(r.shape[0]):
                for j1 in range(r.shape[1]):
                    chm[i1 + i - n0, j1 + j - n0] = max(tree_map[i1, j1], chm[i1 + i - n0, j1 + j - n0])
            
        return chm


@jit(nopython=False)
def get_tree_height(r, a, b, c, height, crown_length, beta0, dx=1.):

    r = r.copy()

    # z position of max radius
    z_max =  (a - 1.) / (a + b - 2.)
    # max tree radius
    r_max = crown_length * c * z_max**(a - 1.) * (1. - z_max)**(b - 1)

    # Initial guess for newton iteration
    z0 = np.ones_like(r)*0.7

    # Newton iteration to find max tree height at given radius
    for k in range(5):
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                r0 =  crown_length * c * z0[i,j]**(a - 1.) * (1. - z0[i,j])**(b - 1) - min(r[i,j],r_max)
                dr_dz = crown_length * c * z0[i,j]**(a - 1) * (-(1.-z0[i,j])**(b-2))*(a*(z0[i,j]-1.) 
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


r = get_synthetic_chm(synthetic_chm, tree_indexes, tree_params)
plt.imshow(r)
plt.colorbar()
plt.show()