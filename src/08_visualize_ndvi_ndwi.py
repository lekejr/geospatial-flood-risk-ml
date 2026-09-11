"""
Step 9: Visualize NDVI and NDWI rasters for Lagos to confirm the
exports look correct.
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt

def plot_index(filepath, title, cmap, output_path, cbar_label):
    with rasterio.open(filepath) as src:
        data = src.read(1)
        if src.nodata is not None:
            data = data.astype(float)
            data[data == src.nodata] = np.nan

    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(data, cmap=cmap, vmin=-1, vmax=1)
    ax.set_title(title)
    ax.axis('off')
    cbar = plt.colorbar(img, ax=ax, fraction=0.036, pad=0.04)
    cbar.set_label(cbar_label)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    print(f"Data range: min={np.nanmin(data):.3f}, max={np.nanmax(data):.3f}, mean={np.nanmean(data):.3f}")
    plt.show()

# NDVI map
plot_index(
    "data/raw/lagos_ndvi.tif",
    "Lagos — NDVI (Vegetation Index)",
    cmap="RdYlGn",
    output_path="outputs/maps/lagos_ndvi.png",
    cbar_label="NDVI"
)

# NDWI map
plot_index(
    "data/raw/lagos_ndwi.tif",
    "Lagos — NDWI (Water Index)",
    cmap="Blues",
    output_path="outputs/maps/lagos_ndwi.png",
    cbar_label="NDWI"
)