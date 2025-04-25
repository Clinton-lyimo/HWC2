import os
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from scipy.stats import gaussian_kde
import rasterio
from rasterio.transform import from_bounds
from libpysal.weights import DistanceBand
from esda import G_Local, Moran_Local
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---- Load Cleaned HWC Data ----
data_dir = "D:/HWC/"
path = os.path.join(data_dir, "CLEANED_Land_for_Life_HWC.csv")
df = pd.read_csv(path)

# Drop rows with missing coordinates
df = df.dropna(subset=['latitude', 'longitude'])

# Aggregate incidents by unique latitude and longitude to get the count of incidents at each location
df_agg = df.groupby(['latitude', 'longitude']).size().reset_index(name='incident_count')

# Convert to GeoDataFrame for spatial analysis
gdf = gpd.GeoDataFrame(
    df_agg,
    geometry=gpd.points_from_xy(df_agg['longitude'], df_agg['latitude']),
    crs="EPSG:4326"
)

# ---- Load Area of Interest (AOI) Shapefile ----
aoi_path = "D:/FILES/shps/AOI/AOI.shp"
if not os.path.exists(aoi_path):
    raise FileNotFoundError(f"Shapefile not found at {aoi_path}.")
aoi_gdf = gpd.read_file(aoi_path)

# Verify AOI CRS
print(f"AOI CRS: {aoi_gdf.crs}")

# Reproject AOI to a projected CRS (UTM zone for Tanzania/Kenya, e.g., UTM Zone 37S, EPSG:32737)
# This avoids issues with centroid calculation in a geographic CRS
aoi_gdf_projected = aoi_gdf.to_crs(epsg=32737)

# Calculate the centroid in the projected CRS
aoi_centroid_projected = aoi_gdf_projected.geometry.centroid.iloc[0]

# Transform the centroid back to EPSG:4326 for Folium
aoi_centroid_gdf = gpd.GeoDataFrame(geometry=[aoi_centroid_projected], crs="EPSG:32737")
aoi_centroid = aoi_centroid_gdf.to_crs("EPSG:4326").geometry.iloc[0]
map_center = [aoi_centroid.y, aoi_centroid.x]


# ---- Fig 2: KDE Map ----
def generate_kde_map():
    # Initialize Folium map
    m = folium.Map(location=map_center, zoom_start=10, tiles="OpenStreetMap")

    # Add AOI boundary
    folium.GeoJson(
        aoi_gdf,
        style_function=lambda x: {'fillColor': 'none', 'color': 'blue', 'weight': 2}
    ).add_to(m)

    # Prepare data for HeatMap with weights based on incident count
    heat_data = [[row["latitude"], row["longitude"], row["incident_count"]] for _, row in df_agg.iterrows()]
    HeatMap(heat_data, radius=15, blur=12, min_opacity=0.5).add_to(m)

    # Save interactive heatmap
    output_file = os.path.join(data_dir, "fig2_kde_map.html")
    m.save(output_file)
    print(f"Fig 2: KDE Map saved as {output_file}")

    # Export KDE as GeoTIFF
    x = df_agg['longitude'].values
    y = df_agg['latitude'].values
    weights = df_agg['incident_count'].values
    x_weighted = np.repeat(x, weights)
    y_weighted = np.repeat(y, weights)
    coords = np.vstack([x_weighted, y_weighted])

    kde = gaussian_kde(coords, bw_method='scott')
    x_min, x_max = x.min() - 0.1, x.max() + 0.1
    y_min, y_max = y.min() - 0.1, y.max() + 0.1
    x_grid, y_grid = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    kde_values = kde(positions).reshape(x_grid.shape)

    output_raster = os.path.join(data_dir, "fig2_kde_map.tif")
    height, width = x_grid.shape
    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)

    with rasterio.open(
            output_raster, 'w', driver='GTiff', height=height, width=width, count=1,
            dtype=kde_values.dtype, crs="EPSG:4326", transform=transform
    ) as dst:
        dst.write(kde_values, 1)
    print(f"Fig 2: KDE GeoTIFF saved as {output_raster}")


# ---- Fig 3: Getis-Ord Gi* Hotspot Map ----
def generate_getis_ord_map():
    # Create spatial weights matrix using a distance threshold (e.g., 0.1 degrees ~ 11 km)
    w = DistanceBand.from_dataframe(gdf, threshold=0.1, binary=True)

    # Calculate Getis-Ord Gi* statistic
    gi = G_Local(gdf['incident_count'].values, w, star=True)
    gdf['Gi_Z'] = gi.Zs  # Z-scores
    gdf['Gi_P'] = gi.p_sim  # p-values

    # Export Z-scores as GeoTIFF
    x = gdf['longitude'].values
    y = gdf['latitude'].values
    z_scores = gdf['Gi_Z'].values

    x_min, x_max = x.min() - 0.1, x.max() + 0.1
    y_min, y_max = y.min() - 0.1, y.max() + 0.1
    x_grid, y_grid = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    grid_shape = x_grid.shape
    z_grid = np.zeros(grid_shape)

    # Interpolate Z-scores onto the grid using nearest neighbor
    for i in range(len(x)):
        col = int((x[i] - x_min) / (x_max - x_min) * (grid_shape[1] - 1))
        row = int((y[i] - y_min) / (y_max - y_min) * (grid_shape[0] - 1))
        z_grid[row, col] = z_scores[i]

    output_raster = os.path.join(data_dir, "fig3_getis_ord_map.tif")
    transform = from_bounds(x_min, y_min, x_max, y_max, grid_shape[1], grid_shape[0])

    with rasterio.open(
            output_raster, 'w', driver='GTiff', height=grid_shape[0], width=grid_shape[1], count=1,
            dtype=z_grid.dtype, crs="EPSG:4326", transform=transform
    ) as dst:
        dst.write(z_grid, 1)
    print(f"Fig 3: Getis-Ord Gi* Map (Z-scores) saved as {output_raster}")


# ---- Fig 4: LISA Cluster-Outlier Map ----
def generate_lisa_map():
    # Create spatial weights matrix
    w = DistanceBand.from_dataframe(gdf, threshold=0.1, binary=True)

    # Calculate LISA (Moran Local)
    moran_loc = Moran_Local(gdf['incident_count'].values, w)
    gdf['Moran_I'] = moran_loc.Is
    gdf['Moran_P'] = moran_loc.p_sim

    # Classify clusters and outliers based on significance (p < 0.05) and quadrant
    cluster_labels = np.zeros(len(gdf), dtype=int)  # 0: Not significant
    for i in range(len(gdf)):
        if moran_loc.p_sim[i] < 0.05:  # Statistically significant
            quad = moran_loc.q[i]
            if quad == 1:
                cluster_labels[i] = 1  # High-High (HH)
            elif quad == 2:
                cluster_labels[i] = 2  # Low-High (LH)
            elif quad == 3:
                cluster_labels[i] = 3  # Low-Low (LL)
            elif quad == 4:
                cluster_labels[i] = 4  # High-Low (HL)

    gdf['Cluster'] = cluster_labels

    # Export cluster labels as GeoTIFF
    x = gdf['longitude'].values
    y = gdf['latitude'].values
    clusters = gdf['Cluster'].values

    x_min, x_max = x.min() - 0.1, x.max() + 0.1
    y_min, y_max = y.min() - 0.1, y.max() + 0.1
    x_grid, y_grid = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    grid_shape = x_grid.shape
    cluster_grid = np.zeros(grid_shape)

    # Interpolate cluster labels onto the grid
    for i in range(len(x)):
        col = int((x[i] - x_min) / (x_max - x_min) * (grid_shape[1] - 1))
        row = int((y[i] - y_min) / (y_max - y_min) * (grid_shape[0] - 1))
        cluster_grid[row, col] = clusters[i]

    output_raster = os.path.join(data_dir, "fig4_lisa_map.tif")
    transform = from_bounds(x_min, y_min, x_max, y_max, grid_shape[1], grid_shape[0])

    with rasterio.open(
            output_raster, 'w', driver='GTiff', height=grid_shape[0], width=grid_shape[1], count=1,
            dtype=cluster_grid.dtype, crs="EPSG:4326", transform=transform
    ) as dst:
        dst.write(cluster_grid, 1)
    print(f"Fig 4: LISA Cluster-Outlier Map saved as {output_raster}")


# ---- Generate All Figures ----
if __name__ == "__main__":
    print(f"Total unique locations: {len(df_agg)}, Total incidents: {df_agg['incident_count'].sum()}")
    generate_kde_map()  # Fig 2
    generate_getis_ord_map()  # Fig 3
    generate_lisa_map()  # Fig 4