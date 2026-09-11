"""
Step 14: Combine all raw layers (elevation, slope, rainfall, NDVI,
NDWI, land cover, flood occurrence) into a single aligned feature
table (CSV), one row per pixel. All layers are resampled onto a
common grid matching the flood occurrence raster's resolution and
extent, since that is our target variable and must not be distorted.
"""

import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import pandas as pd

# Reference raster: flood occurrence (defines our target grid)
reference_path = "data/raw/lagos_flood_occurrence.tif"

with rasterio.open(reference_path) as ref:
    ref_meta = ref.meta.copy()
    ref_shape = (ref.height, ref.width)
    ref_transform = ref.transform
    ref_crs = ref.crs
    flood = ref.read(1)

def resample_to_reference(path, resampling_method=Resampling.bilinear):
    """Reproject/resample a raster onto the reference grid."""
    with rasterio.open(path) as src:
        destination = np.zeros(ref_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=resampling_method
        )
    return destination

print("Resampling all layers onto the flood-occurrence grid...")

elevation = resample_to_reference("data/raw/lagos_elevation.tif")
slope = resample_to_reference("data/raw/lagos_slope.tif")
rainfall = resample_to_reference("data/raw/lagos_rainfall_2023.tif")
ndvi = resample_to_reference("data/raw/lagos_ndvi.tif")
ndwi = resample_to_reference("data/raw/lagos_ndwi.tif")
# Land cover is categorical, so use nearest-neighbor resampling
landcover = resample_to_reference("data/raw/lagos_landcover.tif", Resampling.nearest)

print("Building feature table...")

# Flatten all 2D arrays into 1D columns
df = pd.DataFrame({
    "elevation": elevation.flatten(),
    "slope": slope.flatten(),
    "rainfall": rainfall.flatten(),
    "ndvi": ndvi.flatten(),
    "ndwi": ndwi.flatten(),
    "landcover": landcover.flatten(),
    "flooded": flood.flatten()
})

# Add row/col indices so we can map back to spatial coordinates later
rows, cols = np.indices(ref_shape)
df["row"] = rows.flatten()
df["col"] = cols.flatten()

print(f"Total rows before cleaning: {len(df)}")

# Remove rows where landcover is 0 (outside the Lagos boundary /
# outside our valid study area, based on earlier diagnostics)
df_clean = df[df["landcover"] != 0].copy()

# Remove any remaining rows with missing/invalid values in predictors
df_clean = df_clean.dropna()

print(f"Total rows after cleaning: {len(df_clean)}")
print(f"Flood class balance in cleaned data:")
print(df_clean["flooded"].value_counts(normalize=True) * 100)

# Save to CSV
df_clean.to_csv("data/processed/lagos_feature_table.csv", index=False)
print("Saved: data/processed/lagos_feature_table.csv")