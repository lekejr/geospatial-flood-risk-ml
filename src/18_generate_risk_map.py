"""
Step 19: Apply the trained Random Forest model to the FULL Lagos
feature table (not just the test sample) to generate a continuous
flood-risk probability map across the entire study area. This is
the key output deliverable of the project.
"""

import pandas as pd
import numpy as np
import rasterio
import joblib
import matplotlib.pyplot as plt

# Load the full feature table (all ~4.26M pixels, not the subsample)
print("Loading full feature table...")
df = pd.read_csv("data/processed/lagos_feature_table.csv")

FEATURES = ["elevation", "slope", "rainfall", "ndvi", "ndwi", "landcover"]

# Load trained Random Forest model
model = joblib.load("outputs/models/random_forest.pkl")

# Predict flood-risk probability for every pixel
print("Generating predictions for all pixels (this may take a minute)...")
X_full = df[FEATURES]
risk_prob = model.predict_proba(X_full)[:, 1]
df["risk_prob"] = risk_prob

# Reconstruct the 2D grid using row/col indices, matching the
# original flood occurrence raster's shape
with rasterio.open("data/raw/lagos_flood_occurrence.tif") as ref:
    ref_shape = (ref.height, ref.width)
    ref_meta = ref.meta.copy()

risk_grid = np.full(ref_shape, np.nan, dtype=np.float32)
risk_grid[df["row"].values, df["col"].values] = df["risk_prob"].values

# Save as a GeoTIFF (so it's a real, usable geospatial output)
ref_meta.update(dtype='float32', count=1, nodata=np.nan)
with rasterio.open("outputs/maps/lagos_flood_risk_map.tif", "w", **ref_meta) as dst:
    dst.write(risk_grid, 1)
print("Saved: outputs/maps/lagos_flood_risk_map.tif")

# Also save a nicely styled PNG for the report/README
fig, ax = plt.subplots(figsize=(10, 10))
img = ax.imshow(risk_grid, cmap="RdYlBu_r", vmin=0, vmax=1)
ax.set_title("Lagos — Predicted Flood Risk (Random Forest)")
ax.axis('off')
cbar = plt.colorbar(img, ax=ax, fraction=0.036, pad=0.04)
cbar.set_label("Predicted Flood Probability")

plt.savefig("outputs/maps/lagos_flood_risk_map.png", dpi=200, bbox_inches='tight')
print("Saved: outputs/maps/lagos_flood_risk_map.png")
plt.show()

# Print summary stats
print(f"\nRisk probability summary:")
print(f"Mean: {np.nanmean(risk_grid):.3f}")
print(f"Median: {np.nanmedian(risk_grid):.3f}")
print(f"% of area with risk > 0.5: {100*np.nanmean(risk_grid > 0.5):.2f}%")
print(f"% of area with risk > 0.7: {100*np.nanmean(risk_grid > 0.7):.2f}%")