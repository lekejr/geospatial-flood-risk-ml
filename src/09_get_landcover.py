"""
Step 10: Extract land cover for Lagos using ESA WorldCover 10m,
and derive a built-up/impervious surface layer. Built-up areas
increase surface runoff and are a key flood-risk factor.
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

# Load ESA WorldCover 2021 (10m global land cover)
worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().clip(lagos_geom)

# WorldCover class codes:
# 10 = Tree cover, 20 = Shrubland, 30 = Grassland, 40 = Cropland,
# 50 = Built-up, 60 = Bare/sparse vegetation, 70 = Snow/ice,
# 80 = Permanent water bodies, 90 = Herbaceous wetland,
# 95 = Mangroves, 100 = Moss/lichen

# Export full land cover classification (scale increased to 40m to
# stay under Earth Engine's direct-download size limit)
print("Exporting land cover...")
geemap.ee_export_image(
    worldcover,
    filename="data/raw/lagos_landcover.tif",
    scale=40,
    region=lagos_geom,
    file_per_band=False
)
print("Land cover saved to data/raw/lagos_landcover.tif")