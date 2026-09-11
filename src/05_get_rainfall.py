"""
Step 6: Extract annual total rainfall for the Lagos study area using
CHIRPS daily precipitation data, aggregated to a yearly sum. This
represents typical annual rainfall exposure across Lagos.
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

# Load CHIRPS daily precipitation, filter to a recent full year (2023),
# and sum all days to get total annual rainfall
chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
    .filterDate('2023-01-01', '2023-12-31') \
    .filterBounds(lagos_geom)

annual_rainfall = chirps.sum().clip(lagos_geom)

# Export to GeoTIFF
print("Exporting annual rainfall...")
geemap.ee_export_image(
    annual_rainfall,
    filename="data/raw/lagos_rainfall_2023.tif",
    scale=5000,  # CHIRPS native resolution is ~5km, no point using finer scale
    region=lagos_geom,
    file_per_band=False
)
print("Rainfall saved to data/raw/lagos_rainfall_2023.tif")