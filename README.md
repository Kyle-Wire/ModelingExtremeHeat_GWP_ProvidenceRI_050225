# Modeling Extreme Heat Vulnerability in Providence, RI

This repository contains the full spatial analysis, datasets, and code used in the project:

"Modeling Extreme Heat Vulnerability in Providence, Rhode Island: A Multi-Stage Spatial Ensemble for Urban Climate Planning."

This study develops a fine-scale ensemble model of ambient air temperature (AAT) during extreme heat events, using a 30-meter grid across Providence, Rhode Island. The framework integrates spatial statistics, machine learning, and deep learning for high-resolution urban climate forecasting.

Core Methods
--OLS Regression: Baseline global model for urban heat prediction

--Geographically Weighted Elastic Net (GWEN): Spatially adaptive variable selection

--Laplacian Eigenmaps: Spectral spatial features capturing multiscale spatial autocorrelation

Base Models:

--Geographically Weighted Regression (GWR)

--Geographically Weighted Random Forest (GWRF)

--Gaussian Process Generalized Additive Model with Spatially Varying Coefficients (GGP-GAM-SVC)

Meta-Ensemble:

--LightGBM stacking of out-of-fold base model predictions

Final Correction:

--Deep Kriging (Neural Network residual spatial correction)

Validation:

--Spatial Cross-Validation

--Residual diagnostics (Moran’s I, histograms)

It was developed as part of climate resilience planning efforts for urban environments and is openly published for use, critique, or expansion.

---

## Repository Overview

```text
.
├── data/
│   ├── raw/               # Shapefiles: tree canopy, impervious surface, DEM, etc.
│   ├── processed/         # Model performance and predictions
├── notebooks/             # Python & R notebooks for Ensemble
├── outputs/
│   ├── figures/           # Coefficient maps, residuals, graphical abstract
│   ├── tables/            # Tables from paper
├── README.md              # The document that you're reading this in. Hope you know that!
├── LICENSE                # MIT License
├── requirements.txt       # Python package dependencies
├── CONTRIBUTING.md        # Guidelines for contributing or reproducing the work
├── CITATION.cff           # Citation metadata for this repository


### Method Summary

- OLS Regression: Initial global model to establish baseline explanatory power
- PCA: Dimensionality reduction of five built-environment variables into a single intensity index
- GWR: Local model calibrated using tree canopy, built environment intensity, and elevation
- Bootstrapping: 50 runs to assess coefficient uncertainty at each grid cell
- Validation: Moran’s I and residual mapping for spatial diagnostics

All processing was done using Python and GIS software (ArcGIS Pro), with model runs executed in Google Colabs.




#### How to Reproduce

This project was developed in Visual Studio Code. To run the notebooks:

1. Open any `.py` file in `/notebooks/` in VS Code
2. Install dependencies in the first cell:

```python
!pip install pandas numpy geopandas scikit-learn shap lightgbm tensorflow keras libpysal esda mgwr matplotlib



##### Data Sources

- Ambient Air Temperature: CAPA Strategies HeatWatch Campaign
- Tree Canopy Cover: USGS NLCD (2020)
- Elevation (DEM): City of Providence Open Data Portal
- Urban Form Data: City of Providence Open Data Portal
- Green Spaces: City of Providence Open Data Portal & Rhode Island GIS Hub
- Water Bodies: City of Providence Open Data Portal & Rhode Island GIS Hub

All intersected an processed datasets used in modeling are included in the /data/processed/ folder


###### License

"This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 License."


###### Acknowledgments

Special thanks to Carolyn, Xana, and Reynolds, who supported this work in ways beyond modeling capabilities.
This model was developed independently by Kyle Wire and builds upon climate planning research conducted for Brown University.


####### Citations

Please cite this repository as: 

@misc{wire2025heat,
  author       = {Kyle Wire},
  title        = {Modeling Extreme Heat Vulnerability in Providence, RI},
  year         = {2025},
  howpublished = {\url{https://github.com/Kyle-Wire/ModelingExtremeHeat_GWP_ProvidenceRI_050225}},
  note         = {Geographically Weighted Modeling, Ensemble Learning, Urban Heat Forecasting}
}



