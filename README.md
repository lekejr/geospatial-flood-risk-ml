# Geospatial Machine Learning for Flood Risk Mapping in Lagos, Nigeria

A reproducible geospatial machine learning project that predicts flood risk
across Lagos State, Nigeria, using open satellite data and historical flood
records. Built entirely with free, open-access datasets and standard laptop
computing resources.

## Research Question

To what extent can machine learning models, trained on topographic,
hydrological, and land-cover-derived geospatial variables, distinguish
flood-prone from non-flood-prone areas in Lagos, and which environmental
factors contribute most to elevated flood risk?

## Motivation

Flooding is a major recurring hazard in Lagos, driven by its low-lying
coastal topography, extensive lagoon system, and intense seasonal rainfall.
This project explores whether a data-driven, machine learning approach —
using only free, open satellite data — can produce a credible, interpretable
flood risk assessment as a reproducible complement to traditional flood
hazard mapping.

## Study Area

Lagos State, Nigeria — selected for its well-documented flood history,
strong open-data coverage via Google Earth Engine, and low-lying coastal
geography that makes it a scientifically meaningful case study.

## Data

| Variable | Source | Resolution |
|---|---|---|
| Elevation | SRTM 30m DEM (USGS) | 30m |
| Slope | Derived from SRTM | 30m |
| Rainfall (2023 annual) | CHIRPS Daily (UCSB) | ~5km |
| NDVI | Sentinel-2 (ESA Copernicus) | 40m |
| NDWI | Sentinel-2 (ESA Copernicus) | 40m |
| Land cover | ESA WorldCover v200 | 40m |
| Flood occurrence (target) | Global Flood Database (Tellman et al., 2021, *Nature*) | 40m |

All data acquired programmatically via the Google Earth Engine Python API.

## Methodology

1. Acquired and clipped all layers to the Lagos boundary via Earth Engine
2. Resampled all predictors onto a common 40m grid
3. Built a full feature table (4,263,810 pixels; ~10% historically flooded)
4. Stratified subsample (80,000 pixels) for laptop-friendly training
5. **Spatial block train/test split** (20×20 pixel blocks) to avoid spatial
   leakage — a standard pixel-random split would inflate performance by
   placing near-identical neighboring pixels on both sides of the split
6. Trained Logistic Regression (baseline) and Random Forest, both with
   class balancing to address the ~90/10 class imbalance
7. Verified model results were not driven by trivial data leakage (e.g.,
   confirmed permanent water bodies were only 43% flooded, not ~100%)
8. Generated a full-coverage predicted flood-risk map for Lagos

## Models & Results

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 0.917 | **0.970** |
| Precision | 0.548 | **0.821** |
| Recall | **0.956** | 0.897 |
| F1-score | 0.697 | **0.857** |
| ROC-AUC | **0.983** | 0.973 |

**Random Forest** was selected as the final model due to its substantially
better precision/recall balance, producing a more practically usable risk
map with far fewer false positives while still capturing 89.7% of true
flood-prone areas.

### Feature Importance (Random Forest)

| Feature | Importance |
|---|---|
| Elevation | 0.300 |
| Land cover | 0.221 |
| NDVI | 0.167 |
| NDWI | 0.140 |
| Rainfall | 0.130 |
| Slope | 0.043 |

Elevation and land cover were the strongest predictors, consistent with
established physical drivers of urban flooding.

## Key Outputs

- `outputs/maps/lagos_flood_risk_map.png` / `.tif` — final predicted flood
  risk map (probability surface across all of Lagos)
- `outputs/figures/model_comparison.png` — performance comparison
- `outputs/figures/feature_importance.png` — feature importance plot
- `outputs/figures/confusion_matrices.png` — confusion matrices
- `outputs/maps/` — environmental variable maps (elevation, slope,
  rainfall, NDVI, NDWI, land cover, flood occurrence)
- `reports/report.md` — full research report

## Limitations

- Rainfall and vegetation data are more recent (2023–2024) than the flood
  occurrence record (2000–2018); this assumes reasonable temporal stability
  in terrain and broad land cover patterns
- CHIRPS rainfall resolution (~5km) is far coarser than the 40m analysis
  grid, limiting fine-scale rainfall variability
- The spatial block train/test split reduces but does not fully eliminate
  spatial autocorrelation between train and test sets
- No engineered drainage/stormwater infrastructure data was available

See `reports/report.md` for full discussion of limitations and future work.

## Installation & Reproduction

```bash
git git clone https://github.com/lekejr/geospatial-flood-risk-ml.git
cd geospatial-flood-risk-ml
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

You will also need a free Google Earth Engine account and a registered
Cloud project (see https://code.earthengine.google.com/register). Update
the project ID in each script (`ee.Initialize(project='your-project-id')`)
before running.

Run scripts in `src/` in numerical order (01 through 18) to reproduce the
full pipeline from raw data acquisition through to the final risk map.

## Technology

Python · Google Earth Engine · GeoPandas · Rasterio · Scikit-learn ·
Pandas · NumPy · Matplotlib

## Author

John Leke — B.Sc. Surveying and Geoinformatics, University of Ilorin.
Built as part of a portfolio transitioning from Surveying/GIS toward
Geospatial Data Science and Geospatial AI.