# import pandas as pd
# import os
#
# # ---- Define Paths ----
# data_dir = "D:/HWC/"  # <— Replace this with your folder path
# hwc_path = os.path.join(data_dir, "Land_for_Life_HWC.csv")
# wellbeing_path = os.path.join(data_dir, "LakeNatron_Wellbeing 2022-2024.csv")
#
# # ---- Load Raw Data ----
# hwc = pd.read_csv(hwc_path, encoding="ISO-8859-1", low_memory=False)
# wellbeing = pd.read_csv(wellbeing_path, encoding="ISO-8859-1", low_memory=False)
#
# # ---- Drop Empty and Very Sparse Columns ----
# hwc = hwc.dropna(axis=1, how='all')
# wellbeing = wellbeing.dropna(axis=1, how='all')
#
# hwc = hwc.loc[:, hwc.isnull().mean() < 0.9]
# wellbeing = wellbeing.loc[:, wellbeing.isnull().mean() < 0.9]
#
# # ---- Rename Key Columns ----
# hwc = hwc.rename(columns={
#     "2.1 Tarehe ya Tukio": "date",
#     "Please take coordinates (latitude)": "latitude",
#     "Please take coordinates (longitude)": "longitude"
# })
#
# wellbeing = wellbeing.rename(columns={
#     "Please take coordinates (latitude)": "latitude",
#     "Please take coordinates (longitude)": "longitude",
#     "Community name / group ranch": "community_name",
#     "Landscape": "landscape"
# })
#
# # ---- Convert Dates (if present) ----
# if "date" in hwc.columns:
#     hwc["date"] = pd.to_datetime(hwc["date"], errors="coerce", dayfirst=True)
#
# # ---- Save Cleaned Files ----
# hwc.to_csv(os.path.join(data_dir, "CLEANED_Land_for_Life_HWC.csv"), index=False)
# wellbeing.to_csv(os.path.join(data_dir, "CLEANED_LakeNatron_Wellbeing.csv"), index=False)
#
# print("Cleaned files saved to:", data_dir)



# PROJECT




# //////////////START//////////////////

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Path to your cleaned data
data_path = "D:/HWC/CLEANED_Land_for_Life_HWC.csv"

# Load data
df = pd.read_csv(data_path)

# Drop rows with missing coordinates
df = df.dropna(subset=['latitude', 'longitude'])

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

# Preview
gdf.head()


import geopandas as gpd
import libpysal
from esda.getisord import G_Local
import matplotlib.pyplot as plt
import os

# ---- Step 1: Project to Metric CRS ----
# Project the GeoDataFrame to UTM for distance-based analysis
gdf_proj = gdf.to_crs(epsg=32736)  # Adjust the EPSG code if needed (Tanzania UTM Zone)

# ---- Step 2: Create Weights Matrix (K-Nearest Neighbors) ----
# Create KNN weights matrix
w = libpysal.weights.KNN.from_dataframe(gdf_proj, k=8)
w.transform = 'R'  # Row-standardized weights

# Check for disconnected components
print("Disconnected components (islands):", w.islands)

# ---- Step 3: Run Getis-Ord Gi* ----
# Use a relevant variable if available; for now, using geometry.centroid.x
gi_star = G_Local(gdf_proj.geometry.centroid.x, w)

# Add results to the GeoDataFrame
gdf_proj['gi_star_z'] = gi_star.Zs  # Z-scores
gdf_proj['gi_star_p'] = gi_star.p_sim  # P-values (for significance)

# ---- Step 4: Visualize Hotspots (Static) ----
# Define output directory for static maps
output_dir = "D:/PythonProject/KDE/"
os.makedirs(output_dir, exist_ok=True)

# Plot static map
fig, ax = plt.subplots(figsize=(10, 8))
gdf_proj.plot(column='gi_star_z', cmap='RdYlBu_r', legend=True, edgecolor='black', linewidth=0.5, ax=ax)
ax.set_title("Getis-Ord Gi* Z-Scores – Hotspot Analysis", fontsize=14)
ax.set_xlabel("UTM Easting")
ax.set_ylabel("UTM Northing")
plt.tight_layout()

# Save static map
output_file_static = os.path.join(output_dir, "getis_ord_hotspots.png")
plt.savefig(output_file_static, dpi=300)
print(f"Static map saved to: {output_file_static}")

# Show plot
plt.show()







# //////////LISA////////////

from esda.moran import Moran_Local
import libpysal
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import os

# ---- Step 1: Project Data to Metric CRS ----
# Project the GeoDataFrame to UTM Zone (adjust EPSG code for the region)
gdf_proj = gdf.to_crs(epsg=32736)  # Example: UTM zone for Tanzania (adjust as needed)

# ---- Step 2: Create Spatial Weights ----
# Generate a KNN weights matrix
w_lisa = libpysal.weights.KNN.from_dataframe(gdf_proj, k=8)
w_lisa.transform = 'R'  # Row-standardized weights

# Check for disconnected components
print("Disconnected components (islands):", w_lisa.islands)

# ---- Step 3: Run Moran's Local I ----
# Replace 'geometry'.x with the actual variable of interest if available
moran = Moran_Local(gdf_proj['geometry'].x, w_lisa)

# ---- Step 4: Label Clusters ----
labels = np.array(['Not Significant'] * len(gdf_proj))
sig = moran.p_sim < 0.05  # Significance threshold
quad = moran.q  # Quadrants (1=HH, 2=LH, 3=LL, 4=HL)
labels[sig & (quad == 1)] = 'High-High'
labels[sig & (quad == 2)] = 'Low-High'
labels[sig & (quad == 3)] = 'Low-Low'
labels[sig & (quad == 4)] = 'High-Low'

# Add labels to GeoDataFrame
gdf_proj['lisa_cluster'] = labels

# ---- Step 5: Define Color Mapping ----
# Use contrasting colors for easier differentiation
lisa_colors = {
    'High-High': '#e31a1c',  # Red
    'Low-Low': '#1f78b4',    # Blue
    'Low-High': '#a6cee3',   # Light Blue
    'High-Low': '#fb9a99',   # Light Red
    'Not Significant': '#d9d9d9'  # Grey
}

# ---- Step 6: Create Outputs Directory ----
output_dir = "D:/PythonProject/KDE/"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# ---- Step 7: Visualize LISA Clusters ----
fig, ax = plt.subplots(figsize=(12, 10))
gdf_proj.plot(color=gdf_proj['lisa_cluster'].map(lisa_colors), ax=ax, edgecolor='black', linewidth=0.5)

# Add a legend
for label in lisa_colors:
    ax.scatter([], [], label=label, color=lisa_colors[label], s=50)
ax.legend(title="LISA Cluster Type", loc="lower left", fontsize=10, frameon=True)

# Titles and axis labels
ax.set_title("Local Moran's I – Spatial Autocorrelation of HWC Events", fontsize=16)
ax.set_xlabel("UTM Easting")
ax.set_ylabel("UTM Northing")
ax.axis('equal')

# ---- Step 8: Save High-Quality Map ----
plt.tight_layout()
output_file = os.path.join(output_dir, "lisa_cluster_map.png")
plt.savefig(output_file, dpi=300)  # Save with high resolution
print(f"LISA Cluster Map saved to: {output_file}")
plt.show()






# /////////NOTEBOOK 1B////////////////
