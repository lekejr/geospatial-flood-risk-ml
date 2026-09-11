"""
Step 3: Quick visual sanity check of the Lagos boundary we exported.
Plots the boundary using GeoPandas + Matplotlib and saves a PNG.
"""

import geopandas as gpd
import matplotlib.pyplot as plt

# Load the boundary we saved in Step 2
lagos = gpd.read_file("data/raw/lagos_boundary.geojson")

# Basic info check
print("Boundary CRS:", lagos.crs)
print("Bounding box:", lagos.total_bounds)

# Plot it
fig, ax = plt.subplots(figsize=(8, 8))
lagos.plot(ax=ax, edgecolor='black', facecolor='lightblue')
ax.set_title("Lagos State Boundary — Study Area")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Save the figure
plt.savefig("outputs/maps/lagos_study_area.png", dpi=200, bbox_inches='tight')
print("Map saved to outputs/maps/lagos_study_area.png")

plt.show()