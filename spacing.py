import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray as rio
from scipy.interpolate import Rbf
from scipy.interpolate import LinearNDInterpolator

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
spacing = distances[:,1:].mean(axis=1)


x = x[::1]
y = y[::1]
spacing = spacing[::1]

f = LinearNDInterpolator((x, y), spacing)

xi = np.linspace(x.min(), x.max(), 500)
yi = np.linspace(y.min(), y.max(), 500)
xi, yi = np.meshgrid(xi, yi)

# Interpolate
zi = f(xi, yi)
plt.imshow(zi[::-1,:], vmin=0., vmax=50., extent=(x.min(), x.max(), y.min(), y.max()))
plt.colorbar()


plt.scatter(x, y, c=spacing)
plt.colorbar()

plt.show()


quit()


#plt.subplot(2,1,2)
plt.hist(spacing)
plt.show()