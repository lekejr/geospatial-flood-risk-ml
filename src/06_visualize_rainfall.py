"""
Step 7: Visualize the annual rainfall raster for Lagos to confirm
the export looks correct.
"""

import rasterio
import matplotlib.pyplot as plt

with rasterio.open("data/raw/lagos_rainfall_2023.tif") as src:
    data = src.read(1)
    if src.nodata is not None:
        data = data.astype(float)
        data[data == src.nodata] = float('nan')

fig, ax = plt.subplots(figsize=(8, 8))
img = ax.imshow(data, cmap="Blues")
ax.set_title("Lagos — Total Annual Rainfall (2023)")
ax.axis('off')
cbar = plt.colorbar(img, ax=ax, fraction=0.036, pad=0.04)
cbar.set_label("Rainfall (mm)")

plt.savefig("outputs/maps/lagos_rainfall_2023.png", dpi=200, bbox_inches='tight')
print("Saved: outputs/maps/lagos_rainfall_2023.png")
print(f"Data range: min={data.min():.2f}, max={data.max():.2f}, mean={data.mean():.2f}")
plt.show()