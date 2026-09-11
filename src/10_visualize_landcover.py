"""
Step 11: Visualize the land cover classification for Lagos using
the official ESA WorldCover color scheme, so classes are easy to
interpret (e.g., built-up in red/grey, water in blue, vegetation
in green).
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Official ESA WorldCover class codes, labels, and colors
classes = {
    10: ("Tree cover", "#006400"),
    20: ("Shrubland", "#ffbb22"),
    30: ("Grassland", "#ffff4c"),
    40: ("Cropland", "#f096ff"),
    50: ("Built-up", "#fa0000"),
    60: ("Bare/sparse vegetation", "#b4b4b4"),
    70: ("Snow/ice", "#f0f0f0"),
    80: ("Permanent water bodies", "#0064c8"),
    90: ("Herbaceous wetland", "#0096a0"),
    95: ("Mangroves", "#00cf75"),
    100: ("Moss/lichen", "#fae6a0"),
}

with rasterio.open("data/raw/lagos_landcover.tif") as src:
    data = src.read(1)

# Print class breakdown (% of pixels in each class present in this data)
print("Land cover class breakdown:")
unique, counts = np.unique(data, return_counts=True)
total = counts.sum()
for val, cnt in zip(unique, counts):
    label = classes.get(val, ("Unknown/nodata", "#000000"))[0]
    print(f"  {val} ({label}): {100 * cnt / total:.1f}%")

# Build a colormap matching only the classes present in this raster
present_codes = sorted(unique.tolist())
colors = [classes.get(c, ("Unknown", "#000000"))[1] for c in present_codes]
cmap = ListedColormap(colors)
bounds = present_codes + [present_codes[-1] + 1]
norm = BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(figsize=(8, 8))
img = ax.imshow(data, cmap=cmap, norm=norm)
ax.set_title("Lagos — Land Cover (ESA WorldCover 10m)")
ax.axis('off')

# Build a legend
legend_labels = [classes.get(c, ("Unknown", "#000000"))[0] for c in present_codes]
handles = [plt.Rectangle((0,0),1,1, color=classes.get(c, ("Unknown","#000000"))[1]) for c in present_codes]
ax.legend(handles, legend_labels, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)

plt.savefig("outputs/maps/lagos_landcover.png", dpi=200, bbox_inches='tight')
print("Saved: outputs/maps/lagos_landcover.png")
plt.show()