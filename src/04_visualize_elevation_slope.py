"""
Step 5: Visualize the elevation and slope rasters for Lagos to confirm
they were exported correctly. Produces two PNG maps.
"""

import rasterio
import matplotlib.pyplot as plt

def plot_raster(filepath, title, cmap, output_path, cbar_label):
    with rasterio.open(filepath) as src:
        data = src.read(1)
        # Mask nodata values if present
        if src.nodata is not None:
            data = data.astype(float)
            data[data == src.nodata] = float('nan')

    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(data, cmap=cmap)
    ax.set_title(title)
    ax.axis('off')
    cbar = plt.colorbar(img, ax=ax, fraction=0.036, pad=0.04)
    cbar.set_label(cbar_label)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    print(f"Data range: min={data.min():.2f}, max={data.max():.2f}, mean={data.mean():.2f}")
    plt.show()

# Elevation map
plot_raster(
    "data/raw/lagos_elevation.tif",
    "Lagos — Elevation (SRTM 30m)",
    cmap="terrain",
    output_path="outputs/maps/lagos_elevation.png",
    cbar_label="Elevation (m)"
)

# Slope map
plot_raster(
    "data/raw/lagos_slope.tif",
    "Lagos — Slope",
    cmap="YlOrRd",
    output_path="outputs/maps/lagos_slope.png",
    cbar_label="Slope (degrees)"
)