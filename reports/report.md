# Geospatial Machine Learning for Flood Risk Mapping in Lagos, Nigeria

## 1. Introduction

Flooding is one of the most recurring and disruptive environmental hazards facing
Lagos, Nigeria's largest city and a major economic hub in West Africa. Rapid
urbanization, a low-lying coastal topography, an extensive lagoon system, and
intense seasonal rainfall combine to make large parts of the city vulnerable to
recurrent flood events. Understanding *where* flood risk is concentrated, and
*why*, is essential for planning, infrastructure investment, and disaster
preparedness.

This project applies geospatial machine learning to model flood risk across
Lagos State using open-access, satellite-derived environmental data combined
with a historical, peer-reviewed flood occurrence record. It demonstrates a
complete, reproducible geospatial data science workflow: from raw satellite
and administrative data, through spatial feature engineering, to supervised
machine learning and model interpretation.

## 2. Problem Statement

Traditional flood risk assessment in Nigeria has historically relied on
localized surveys, anecdotal reporting, or coarse regional hazard maps, which
are often difficult to update, expensive to produce, and lack fine spatial
resolution. Meanwhile, freely available satellite Earth observation data
(elevation models, precipitation records, vegetation and water indices, and
land cover classifications) combined with modern machine learning techniques
offer an opportunity to produce data-driven, reproducible, and updatable flood
risk assessments at a fraction of the cost.

This project investigates whether such an approach can produce a credible,
interpretable flood risk model for Lagos using only open, freely accessible
data and modest computational resources — a workflow suitable for
resource-constrained contexts and repeatable for other Nigerian cities in
future work.

## 3. Research Question

**To what extent can machine learning models, trained on topographic,
hydrological, and land-cover-derived geospatial variables, distinguish
flood-prone from non-flood-prone areas in Lagos, and which environmental
factors contribute most to elevated flood risk?**

This question has two components that structure the analysis:

1. **Predictive component** — can supervised classification models
   (Logistic Regression, Random Forest) reliably distinguish historically
   flooded from non-flooded locations using environmental predictors alone?
2. **Explanatory component** — which of these environmental variables
   (elevation, slope, rainfall, vegetation cover, surface wetness, land
   cover) are most influential in driving the model's predictions, and does
   this align with known physical drivers of flooding?
## 4. Study Area

The study area is **Lagos State, Nigeria**, defined using the FAO GAUL 2015
Level 1 administrative boundary dataset. Lagos was selected over other
candidate Nigerian cities (Abuja, Ibadan, Port Harcourt) because it offers
the strongest combination of:

- A well-documented history of flooding, with multiple recorded events
  between 2000 and 2018 in the Global Flood Database used for this project.
- High-quality, freely available satellite coverage through Google Earth
  Engine, requiring no manual download of large datasets.
- A physically distinctive, low-lying coastal and lagoon-fringed
  environment, making it a scientifically meaningful and policy-relevant
  case study.

The Lagos boundary spans approximately longitude 2.71°E to 4.35°E and
latitude 6.37°N to 6.71°N, an elongated coastal geography running from
Badagry in the west to Epe/Lekki in the east.

## 5. Data

All data used in this project are open, credible, and satellite- or
administratively-derived. No proprietary, fabricated, or manually digitized
data were used.

| Variable | Dataset | Source | Resolution |
|---|---|---|---|
| Study area boundary | FAO GAUL 2015, Level 1 | Earth Engine / FAO | Vector |
| Elevation | SRTM 30m DEM | USGS (via Earth Engine) | 30m |
| Slope | Derived from SRTM DEM | Computed in Earth Engine | 30m |
| Rainfall | CHIRPS Daily Precipitation (2023 annual total) | UCSB Climate Hazards Center (via Earth Engine) | ~5km |
| NDVI | Sentinel-2 Surface Reflectance (median composite, Nov 2023–Feb 2024) | ESA Copernicus (via Earth Engine) | 40m (resampled) |
| NDWI | Sentinel-2 Surface Reflectance (median composite, Nov 2023–Feb 2024) | ESA Copernicus (via Earth Engine) | 40m (resampled) |
| Land cover | ESA WorldCover v200 (2021) | ESA (via Earth Engine) | 40m (resampled) |
| Flood occurrence (target variable) | Global Flood Database (Tellman et al., 2021, *Nature*) | Earth Engine, peer-reviewed | 40m (resampled) |

Eight distinct historical flood events intersecting Lagos were identified in
the Global Flood Database and combined into a single binary occurrence
layer: a pixel is labeled `1` if it was recorded as flooded in at least one
event between 2000–2018, and `0` otherwise.

## 6. Methodology

### 6.1 Data Acquisition
All raster and vector data were acquired programmatically via the Google
Earth Engine Python API, clipped to the Lagos boundary, and exported as
local GeoTIFF files. This ensures the entire acquisition process is
scripted and reproducible, rather than relying on manual downloads.

### 6.2 Spatial Alignment
Source layers were captured at different native resolutions (30m for
elevation/slope; ~5km for rainfall; 10–20m for Sentinel-2-derived indices;
10m for land cover). All predictor layers were resampled onto a common
40m grid matching the flood occurrence raster, using bilinear resampling
for continuous variables (elevation, slope, rainfall, NDVI, NDWI) and
nearest-neighbor resampling for the categorical land cover layer, to avoid
introducing artificial class values.

### 6.3 Feature Table Construction
All aligned layers were flattened into a single tabular dataset, one row
per 40m pixel, with six predictor columns (elevation, slope, rainfall,
NDVI, NDWI, land cover) and one binary target column (flood occurrence).
Pixels outside the valid Lagos study area (identified via the land cover
layer's nodata value) were removed. The resulting clean dataset contained
**4,263,810 pixels**, of which **10.02% were historically flooded** and
**89.98% were not** — a realistic and expected class imbalance for a rare
hazard event.

### 6.4 Subsampling
For computational efficiency on standard laptop hardware, a stratified
random subsample of 80,000 pixels was drawn from the full feature table,
preserving the original ~10%/90% class balance.

### 6.5 Spatial Train/Test Split
A naive random pixel-level train/test split would risk **spatial leakage**:
neighboring pixels are highly spatially autocorrelated (near-identical
elevation, rainfall, and land cover), so placing adjacent pixels on both
sides of a split would let the model implicitly "see" test locations during
training, inflating apparent performance.

To address this, the study area was divided into a grid of spatial blocks
(20×20 pixels per block, 10,751 unique blocks total), and entire blocks —
not individual pixels — were randomly assigned to training (80%) or testing
(20%). This ensures no test pixel is spatially adjacent to a training pixel
from the same block, providing a simple but defensible safeguard against
spatial leakage. The resulting split produced 64,005 training pixels and
15,995 test pixels, with class balance closely preserved in both sets
(~10% flooded in each).
## 7. Feature Engineering

Six predictor variables were selected based on their established physical
relevance to flood risk, availability, and technical feasibility. Variables
that were considered but not included (e.g., soil characteristics,
drainage density from vector hydrology data) were excluded where reliable,
easily accessible open data was not available for Lagos at a reasonable
resolution — consistent with this project's aim of using only credible,
practically obtainable data.

| Feature | Why it matters for flood risk |
|---|---|
| **Elevation** | Lower-lying land is physically more prone to water accumulation and slower drainage. |
| **Slope** | Flatter terrain drains poorly, increasing water retention time; steeper terrain sheds water more quickly. |
| **Rainfall** | Total precipitation is a direct driver of surface water volume and runoff. |
| **NDVI** (vegetation index) | Vegetated land generally allows better rainfall infiltration and reduces surface runoff compared to bare or paved land. |
| **NDWI** (water index) | Highlights surface water and saturated soil, acting as a proxy for wetness and proximity to water. |
| **Land cover class** | Distinguishes built-up (impervious, runoff-heavy), vegetated, and water/wetland areas, each with very different hydrological behavior. |

No deep learning or complex feature transformations were used, in line with
the project's goal of demonstrating clear, interpretable spatial feature
engineering appropriate for the available computational resources.

## 8. Machine Learning Models

Two supervised classification models were trained to predict flood
occurrence (binary: flooded / not flooded) from the six environmental
predictors:

1. **Logistic Regression** — a linear baseline model, included to establish
   a simple, interpretable point of comparison.
2. **Random Forest** — an ensemble, non-linear model capable of capturing
   more complex interactions between environmental variables.

Both models were trained using `class_weight="balanced"` in scikit-learn to
explicitly address the ~90/10 class imbalance in the dataset, ensuring the
minority (flooded) class was not effectively ignored during training — a
common failure mode when imbalance is not addressed. Logistic Regression
features were standardized (zero mean, unit variance) prior to training, as
is standard practice for linear models; Random Forest was trained on
unscaled features, as tree-based models are insensitive to feature scale.

Extreme Gradient Boosting (XGBoost) was considered as an optional third
model but was not included, as the Random Forest model already achieved
strong, well-balanced performance (see Section 9), and adding a third
algorithm was judged not to materially strengthen the project's findings
relative to the additional complexity involved.

## 9. Model Evaluation

Models were evaluated on the held-out spatial test set (15,995 pixels,
9.93% flooded) using accuracy, precision, recall, F1-score, ROC-AUC, and
confusion matrices. Because the dataset is imbalanced, accuracy alone is
not a reliable indicator of model quality (a model predicting "not flooded"
for every pixel would achieve ~90% accuracy while being practically
useless); F1-score and ROC-AUC were treated as the primary evaluation
metrics.

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 0.917 | **0.970** |
| Precision | 0.548 | **0.821** |
| Recall | **0.956** | 0.897 |
| F1-score | 0.697 | **0.857** |
| ROC-AUC | **0.983** | 0.973 |

*(See `outputs/figures/model_comparison.png` and
`outputs/figures/confusion_matrices.png`.)*

Random Forest achieved a substantially better precision/recall balance
(F1 = 0.857 vs 0.697), meaning it produced far fewer false-positive flood
predictions while still correctly identifying 89.7% of actual flooded
pixels. Logistic Regression achieved marginally higher recall and ROC-AUC,
but at a considerable cost to precision (0.548), generating many more false
alarms. Given that a practically useful flood risk map should minimize
excessive false positives while still capturing genuine risk areas, Random
Forest was selected as the final model used to generate the project's flood
risk map (Section 10).

### 9.1 Data Leakage Check

Before accepting these results, a specific check was performed to rule out
a trivial form of data leakage: since permanent water bodies are
definitionally "wet," there was a risk that the model could achieve high
performance simply by learning "water = flooded," which would be
physically true but analytically meaningless for identifying flood-prone
*land*. Examining the flood rate by land cover class showed permanent water
bodies were flooded in only 43.1% of cases (not near 100%), and wetlands in
29.0% of cases — confirming the flood label reflects genuine historical
inundation events rather than simply mapping onto permanent water extent.
This supports the validity of the reported model performance.
## 10. Results: Flood Risk Map

The trained Random Forest model was applied to all 4,263,810 pixels across
the full Lagos study area (not just the test sample) to generate a
continuous flood-risk probability surface, where each pixel is assigned a
predicted probability of flooding between 0 and 1.

*(See `outputs/maps/lagos_flood_risk_map.png` and the georeferenced raster
`outputs/maps/lagos_flood_risk_map.tif`.)*

Summary statistics of the predicted risk surface:

| Statistic | Value |
|---|---|
| Mean predicted risk | 0.117 |
| Median predicted risk | 0.000 |
| % of area with risk > 0.5 | 11.16% |
| % of area with risk > 0.7 | 10.00% |

The median risk of 0.000 confirms that the model confidently classifies
the majority of Lagos as low flood risk, consistent with the true base
rate of historical flooding (10.02%). The proportion of the study area
flagged at high risk (>0.5 probability: 11.16%; >0.7 probability: 10.00%)
closely tracks the true historical flood occurrence rate, indicating the
model produces a well-calibrated risk surface rather than systematically
over- or under-predicting risk at the landscape scale. The mean risk
(0.117) sitting slightly above the true flood rate reflects the model
assigning intermediate, non-zero probability to plausible at-risk areas
adjacent to confirmed flood zones — a reasonable and expected behavior for
a probabilistic risk map, rather than a simple reproduction of the binary
training labels.

## 11. Feature Importance

Random Forest feature importances (Gini importance) were extracted to
identify which environmental variables most strongly influenced the
model's predictions:

*(See `outputs/figures/feature_importance.png`.)*

| Feature | Importance |
|---|---|
| Elevation | 0.300 |
| Land cover | 0.221 |
| NDVI | 0.167 |
| NDWI | 0.140 |
| Rainfall | 0.130 |
| Slope | 0.043 |

**Elevation** was the single most influential predictor, consistent with
the fundamental physical relationship between low-lying terrain and flood
susceptibility. **Land cover** was the second most important variable,
reflecting the strong distinction between impervious/built-up surfaces,
vegetated land, and water/wetland classes in shaping local hydrological
behavior. **NDVI and NDWI** contributed meaningfully, reinforcing that
vegetation cover and surface wetness carry real predictive signal beyond
what elevation and land cover alone capture. **Rainfall** contributed less
than the terrain-based variables, plausibly because annual rainfall varies
relatively little across the compact Lagos study area compared to the
sharp local variation in elevation and land cover. **Slope** was the least
important feature, likely reflecting Lagos's overall flat topography (mean
slope of only 1.32° across the study area), which limits its ability to
discriminate between locations.

## 12. Spatial Interpretation

Cross-referencing the flood occurrence rate by land cover class (Section
9.1) with the feature importance ranking provides a coherent physical
narrative: permanent water bodies (43.1% historically flooded) and
herbaceous wetlands (29.0%) show the highest flood rates, aligning with
their low elevation and high NDWI values. Bare/sparse vegetation areas
(10.3% flooded) also show elevated risk, consistent with poorly vegetated,
often low-lying or degraded land offering limited infiltration capacity.
In contrast, built-up areas (0.27% flooded), tree cover (0.18%), and
cropland (0.11%) show markedly lower historical flood rates, suggesting
these areas are disproportionately located on the relatively higher,
better-drained parts of Lagos captured in this dataset.

This pattern is physically coherent and supports the conclusion that the
model has learned a genuine, interpretable relationship between terrain,
land cover, vegetation/wetness, and historical flood occurrence, rather
than an arbitrary or spurious statistical association.
## 13. Limitations

This project has several limitations that should be considered when
interpreting its results:

- **Temporal mismatch between predictors and labels.** The flood
  occurrence record spans 2000–2018, while the rainfall, NDVI, and NDWI
  layers reflect more recent conditions (2023–2024) due to the practical
  availability of open data. This is a simplification: it assumes the
  broad spatial pattern of terrain, land cover, and typical rainfall
  distribution across Lagos has remained reasonably stable, which is a
  reasonable but not perfect assumption for elevation and topography, and
  a weaker one for land cover and vegetation, which can change over
  15–20 years due to urban growth.

- **Coarse resolution of rainfall data.** CHIRPS rainfall data has a native
  resolution of approximately 5km, far coarser than the 40m analysis grid.
  This means rainfall values are effectively uniform across large areas of
  the study region and likely explain why rainfall ranked below
  terrain-based variables in feature importance — the true fine-scale
  rainfall variability across Lagos is not captured by this dataset.

- **Simplified spatial validation.** The 20×20 pixel block-based spatial
  train/test split reduces, but does not fully eliminate, spatial
  autocorrelation effects between training and test sets. More rigorous
  approaches (e.g., spatial k-fold cross-validation with larger buffer
  distances, or held-out geographic sub-regions) could provide a more
  conservative estimate of true generalization performance.

- **Binary flood label simplification.** The target variable treats flood
  occurrence as binary (ever flooded vs. never flooded, 2000–2018) rather
  than capturing flood frequency, severity, or duration, which would offer
  a richer but more complex modeling target.

- **No explicit drainage infrastructure data.** Engineered drainage
  systems, stormwater infrastructure, and canal networks — known to
  significantly influence urban flooding in Lagos — were not available as
  open geospatial data and are therefore not represented in this model.

## 14. Conclusion

This project demonstrates that machine learning models trained on freely
available, satellite-derived geospatial variables — elevation, slope,
rainfall, vegetation index, water index, and land cover — can meaningfully
distinguish historically flood-prone areas from non-flood-prone areas in
Lagos, Nigeria. A Random Forest classifier achieved strong, well-balanced
performance (F1-score = 0.857, ROC-AUC = 0.973) after explicitly addressing
class imbalance and spatial leakage, two methodological considerations
central to responsible geospatial machine learning. Elevation and land
cover emerged as the strongest predictors, consistent with established
physical understanding of urban flood dynamics, lending credibility to the
model's learned relationships beyond raw predictive accuracy alone.

The resulting flood risk map provides a reproducible, updatable,
data-driven complement to traditional flood hazard assessment approaches,
built entirely from open data and standard laptop-level computing
resources — demonstrating that credible geospatial AI analysis does not
require expensive infrastructure or proprietary datasets.

## 15. Future Work

Several directions could extend this project:

- Incorporating higher-resolution or more temporally-matched rainfall data
  (e.g., satellite-derived daily precipitation at finer spatial resolution)
  to better capture local rainfall variability.
- Extending the binary flood label to a multi-class or continuous flood
  frequency/severity target, if suitable historical data becomes available.
- Applying more rigorous spatial cross-validation techniques (e.g.,
  spatial k-fold with explicit buffer zones) to more conservatively assess
  generalization performance.
- Extending this workflow to other Nigerian cities (e.g., Port Harcourt,
  Ibadan) to test whether the same modeling approach and feature set
  generalize across different urban and topographic contexts.
- Incorporating drainage infrastructure or urban stormwater network data,
  where available, to capture engineered flood mitigation factors not
  reflected in natural environmental variables alone.
  