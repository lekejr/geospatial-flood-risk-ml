"""
Step 4: Extract elevation (SRTM DEM) and derive slope for the Lagos
study area, clipped to the boundary from Step 2. Exports both as
GeoTIFF rasters for use in feature engineering later.
"""

import ee
import geemap
import geopandas as gpd

# Initialize Earth Engine
ee.Initialize(project='lagos-flood-risk-ml')

# Load our Lagos boundary (from Step 2) and convert to an Earth Engine geometry
lagos_gdf = gpd.read_file("data/raw/lagos_boundary.geojson")
lagos_geojson = lagos_gdf.geometry.iloc[0].__geo_interface__
lagos_geom = ee.Geometry(lagos_geojson)

# Load SRTM 30m Digital Elevation Model
dem = ee.Image("USGS/SRTMGL1_003").clip(lagos_geom)

# Compute slope (in degrees) from the DEM
slope = ee.Terrain.slope(dem)

# Export elevation to a local GeoTIFF
print("Exporting elevation...")
geemap.ee_export_image(
    dem,
    filename="data/raw/lagos_elevation.tif",
    scale=30,
    region=lagos_geom,
    file_per_band=False
)
print("Elevation saved to data/raw/lagos_elevation.tif")

# Export slope to a local GeoTIFF
print("Exporting slope...")
geemap.ee_export_image(
    slope,
    filename="data/raw/lagos_slope.tif",
    scale=30,
    region=lagos_geom,
    file_per_band=False
)
print("Slope saved to data/raw/lagos_slope.tif")