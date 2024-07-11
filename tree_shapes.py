import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray as rio
import fastfuels_core
import tqdm


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

# Create a Tree Population object to represent the tree inventory and for easy access to tree data
tree_population = fastfuels_core.TreePopulation(tree_inventory_gdf_3857)

# Populate the synthetic CHM with tree heights
for tree in tqdm(tree_population):
    if tree.status_code != 1:
        continue
    crown_radius = tree.max_crown_radius
    #crown = Point(tree.x, tree.y).buffer(crown_radius)
    
    print(tree)