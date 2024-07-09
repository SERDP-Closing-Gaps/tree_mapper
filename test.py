import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray as rio


data = pd.read_csv('data/tree_inventory.csv')
tree_inventory = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data["X"], data["Y"]), crs="EPSG:4326")
tree_inventory = tree_inventory.to_crs(3857)
tree_inventory["X"] = tree_inventory.geometry.x
tree_inventory["Y"] = tree_inventory.geometry.y
tree_inventory = tree_inventory.dropna()

x = tree_inventory['X'].to_numpy()
y = tree_inventory['Y'].to_numpy()
coords = np.c_[x, y]
h = tree_inventory['HT'].to_numpy()

h_min = h.min()
h_max = h.max()

print(len(h))
quit()


chm_data = rio.open_rasterio('data/chm.tif')
z = chm_data.data[0]
xi, yi = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
z = z.flatten()
xi = xi.flatten()
yi = yi.flatten()

indexes = np.argsort(z)
z = z[indexes]
xi = xi[indexes]
yi = yi[indexes]

indexes = np.logical_and(z <= h_max + 0.1, z >= h_min - 0.1)

z = z[indexes]
xi = xi[indexes]
yi = yi[indexes]

bins = np.arange(h_min, h_max, 2.5)


print(bins)
quit()

#indexes = np.searchsorted(z, [2.5, ,15.,20.,25.,30.])
print(indexes)

quit()

zs = chm.data[0].flatten()
zs = zs[zs > 0.5]

plt.subplot(2,1,1)
plt.hist(zs, bins=np.linspace(1.,40.,41))








nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(coords)
distances, indices = nbrs.kneighbors(coords)

heights = tree_inventory['HT'].to_numpy()

print(heights.min())
quit()

plt.subplot(2,1,2)
plt.hist(heights, bins=np.linspace(1.,40.,41))
plt.show()

quit()

plt.subplot(2,1,2)
plt.hist(distances[:,1], bins=np.linspace(0.,10., 100))
plt.show()