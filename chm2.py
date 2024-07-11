import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray as rio
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label
from numba import jit, types
import numpy.random as random
from numba.typed import List
from scipy.spatial import KDTree
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from scipy.ndimage import distance_transform_edt

# CHM Data
chm_data = rio.open_rasterio('data/chm.tif')
chm = chm_data.data[0]
xx, yy = np.meshgrid(chm_data.x, chm_data.y)

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

def get_seed_points(mask):
    labeled_mask, num_features = label(mask)

    component_points = []
    for label_num in range(1, num_features + 1):
        coords = np.argwhere(labeled_mask == label_num)
        if coords.size > 0:
            rand_index = np.random.randint(0, len(coords))
            component_points.append((coords[rand_index][0], coords[rand_index][1]))  

    return component_points

# Try different seed points

from skimage.feature import peak_local_max

thing = peak_local_max(chm, min_distance=5, threshold_abs = 1.)

mask = np.zeros_like(chm)
mask[thing[:,0], thing[:,1]] = 1.
plt.imshow(mask)
plt.show()
quit()

d = distance_transform_edt(mask==0)
print(d.shape)
plt.imshow(d)
plt.show()


quit()



def sample_trees(chm_data, radius):


    # Slightly smooth the chm data?
    #chm_data = gaussian_filter(chm_data, sigma=1.)
    # Tree mask
    mask = np.zeros_like(chm_data)
    mask[chm_data > 1.] = 1.

    # Get seed points distributed over different tree clusters
    active = get_seed_points(mask)
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

        return marked, active


    @jit(nopython=True)
    def get_point(i, j, radius, dx):
        # Randomly generate a new point nearby point i,j
        px = i*dx 
        py = j*dx
        r = radius[i,j]*(random.random() + 1.)
        theta =  2. * np.pi * random.random()
        px1 = px + r*np.cos(theta)
        py1 = py + r*np.sin(theta)
        i1 = int(px1 / dx)
        j1 = int(py1 / dx)

        return (i1, j1)


    @jit(nopython=True)
    def is_valid(xx, yy, marked, mask, i, j, radius, dx):

        # Check to make sure point is in bounds 
        n0 = marked.shape[0]
        n1 = marked.shape[1]
        valid = (1 <= i and i <= n0-1 and 1 <= j and j <= n1-1 and mask[i,j] > 0.)

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


    marked, active = disk_sample(xx, yy, marked, mask, active, radius, dx=dx)
    
    print(marked, active)

    plt.subplot(3,1,1)
    plt.imshow(marked)

    synthetic = np.zeros_like(chm_data)
    indexes = marked > 0.
    synthetic[indexes] = chm_data[indexes]
    print(marked.sum())

    synthetic = gaussian_filter(synthetic, sigma=1.)

    plt.subplot(3,1,2)
    plt.imshow(chm_data)
    plt.colorbar()

    plt.subplot(3,1,3)
    plt.imshow(synthetic)
    plt.colorbar()

    plt.show()

    hs = chm_data[indexes]
    plt.hist(hs, bins=100)
    plt.show()

sample_trees(chm, radius)



