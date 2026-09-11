"""
Step 2: Define and export the Lagos study area boundary.
This uses the FAO GAUL administrative boundaries dataset (a standard,
reputable, free dataset available directly in Earth Engine) to extract
the Lagos State boundary, which we'll use to clip all other geospatial
variables in later steps.
"""

import ee
import geemap

# Initialize Earth Engine with your project
ee.Initialize(project='lagos-flood-risk-ml')

# Load the FAO GAUL Level 1 administrative boundaries (states/provinces)
admin_boundaries = ee.FeatureCollection("FAO/GAUL/2015/level1")

# Filter to Nigeria, then to Lagos State
lagos = admin_boundaries.filter(
    ee.Filter.And(
        ee.Filter.eq('ADM0_NAME', 'Nigeria'),
        ee.Filter.eq('ADM1_NAME', 'Lagos')
    )
)

# Check that we found exactly one feature (Lagos)
count = lagos.size().getInfo()
print(f"Number of matching features found: {count}")

if count == 0:
    print("ERROR: Lagos boundary not found. Check the filter names.")
else:
    # Get basic info to confirm we have the right area
    lagos_info = lagos.first().getInfo()
    print("Successfully found Lagos boundary.")
    print(f"Properties: {lagos_info['properties']}")

    # Export the boundary as a GeoJSON file to data/raw/
    geemap.ee_export_vector(lagos, filename="data/raw/lagos_boundary.geojson")
    print("Lagos boundary saved to data/raw/lagos_boundary.geojson")