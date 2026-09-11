"""
Step 12: Extract historical flood occurrence for Lagos using the
Global Flood Database (Tellman et al., 2021, Nature) — a peer-reviewed,
satellite-derived flood event dataset. We combine all recorded flood
events intersecting Lagos into a single binary layer:
1 = flooded at least once in the record, 0 = never recorded as flooded.
This becomes our machine-learning target variable.
"""

import ee
import geemap
import geopandas as gpd

# Initialize Earth Engine
ee.Initialize(project='lagos-flood-risk-ml')

# Load Lagos boundary
lagos_gdf = gpd.read_file("data/raw/lagos_boundary.geojson")
lagos_geojson = lagos_gdf.geometry.iloc[0].__geo_interface__
lagos_geom = ee.Geometry(lagos_geojson)

# Load the Global Flood Database (MODIS-based, covers 2000-2018)
gfd = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1") \
    .filterBounds(lagos_geom)

# Check how many flood events intersect Lagos
event_count = gfd.size().getInfo()
print(f"Number of recorded flood events intersecting Lagos: {event_count}")

if event_count == 0:
    print("WARNING: No flood events found for this region in the database.")
else:
    # Combine all events: a pixel is '1' if it was flooded in ANY event
    flood_binary = gfd.select('flooded').max().clip(lagos_geom)

    print("Exporting flood occurrence layer...")
    geemap.ee_export_image(
        flood_binary,
        filename="data/raw/lagos_flood_occurrence.tif",
        scale=40,
        region=lagos_geom,
        file_per_band=False
    )
    print("Flood occurrence saved to data/raw/lagos_flood_occurrence.tif")