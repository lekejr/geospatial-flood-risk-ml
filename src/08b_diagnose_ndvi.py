"""
Diagnostic: Investigate why NDVI/NDWI rasters are showing all-NaN.
"""

import rasterio
import numpy as np

for name, path in [("NDVI", "data/raw/lagos_ndvi.tif"), ("NDWI", "data/raw/lagos_ndwi.tif")]:
    print(f"\n--- {name} ---")
    with rasterio.open(path) as src:
        data = src.read(1)
        print("Shape:", data.shape)
        print("Dtype:", data.dtype)
        print("Declared nodata value:", src.nodata)
        print("Raw min/max (no masking):", np.nanmin(data), np.nanmax(data))
        print("Count of NaN pixels:", np.isnan(data).sum(), "out of", data.size)
        print("Count of zero pixels:", (data == 0).sum())
        print("Sample values (first 5x5 block):")
        print(data[:5, :5])