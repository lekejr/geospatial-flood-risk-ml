"""
Step 13: Visualize the flood occurrence layer for Lagos and report
the class balance (% flooded vs not flooded).

Note: This raster's declared nodata value is 0, which is also our
legitimate "not flooded" class value. We ignore the nodata flag here
since the diagnostic confirmed only valid 0/1 values are present.
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

with rasterio.open("data/raw/lagos_flood_occurrence.tif") as src:
    data = src.read(1)

flooded = (data == 1).sum()
not_flooded = (data == 0).sum()
total = flooded + not_flooded

print(f"Total pixels: {total}")
print(f"Flooded pixels: {flooded} ({100*flooded/total:.2f}%)")
print(f"Not flooded pixels: {not_flooded} ({100*not_flooded/total:.2f}%)")

cmap = ListedColormap(["#e0e0e0", "#0033cc"])  # grey = not flooded, blue = flooded

fig, ax = plt.subplots(figsize=(8, 8))
img = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
ax.set_title("Lagos — Historical Flood Occurrence (2000-2018)")
ax.axis('off')

handles = [plt.Rectangle((0,0),1,1, color="#e0e0e0"), plt.Rectangle((0,0),1,1, color="#0033cc")]
ax.legend(handles, ["Never flooded (recorded)", "Flooded at least once"], loc='upper left', bbox_to_anchor=(1.02, 1))

plt.savefig("outputs/maps/lagos_flood_occurrence.png", dpi=200, bbox_inches='tight')
print("Saved: outputs/maps/lagos_flood_occurrence.png")
plt.show()