"""
Step 8: Compute NDVI and NDWI for Lagos using a cloud-filtered Sentinel-2
median composite. NDVI indicates vegetation health/cover (affects runoff
and infiltration). NDWI highlights surface water and saturated soil
(a proxy for flood-prone areas).
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

# Load Sentinel-2 Surface Reflectance, filter by date, location, and cloud cover
s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
    .filterDate('2023-11-01', '2024-02-28') \
    .filterBounds(lagos_geom) \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

# Take the median composite (reduces cloud/noise effects)
composite = s2.median().clip(lagos_geom)

# Compute NDVI: (NIR - Red) / (NIR + Red) -> bands B8, B4
ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI')

# Compute NDWI: (Green - NIR) / (Green + NIR) -> bands B3, B8
ndwi = composite.normalizedDifference(['B3', 'B8']).rename('NDWI')

# Export NDVI (scale increased to 40m to stay under Earth Engine's
# direct-download size limit)
print("Exporting NDVI...")
geemap.ee_export_image(
    ndvi,
    filename="data/raw/lagos_ndvi.tif",
    scale=40,
    region=lagos_geom,
    file_per_band=False
)
print("NDVI saved to data/raw/lagos_ndvi.tif")

# Export NDWI
print("Exporting NDWI...")
geemap.ee_export_image(
    ndwi,
    filename="data/raw/lagos_ndwi.tif",
    scale=40,
    region=lagos_geom,
    file_per_band=False
)
print("NDWI saved to data/raw/lagos_ndwi.tif")