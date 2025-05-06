# Modeling Extreme Heat Vulnerability in Providence, RI

This repository contains the full spatial analysis, datasets, and code used in the project: 
 
**"Modeling Extreme Heat Vulnerability in Providence, Rhode Island: A Multi-Scale Spatial Analysis Around Brown University."**

This study develops a fine-scale spatial model of ambient air temperature (AAT) during extreme heat events, using a 30-meter analysis grid over Providence, Rhode Island. The project combines:

- Principal Component Analysis (PCA) to reduce multicollinearity across urban form variables
- Geographically Weighted Regression (GWR) to model spatially varying relationships between land surface characteristics and temperature
- Bootstrapping to evaluate model uncertainty and local coefficient stability

It was developed as part of climate resilience planning efforts for urban environments and is openly published for use, critique, or expansion.

---

## Repository Overview

```text
.
├── data/
│   ├── raw/               # Intersected shapefiles: tree canopy, impervious surface, DEM, etc.
│   ├── processed/         # Grid-level AAT dataset, PCA results, GWR bootstraps
├── notebooks/             # Python notebooks for GWR & bootstrapping
├── outputs/
│   ├── figures/           # Coefficient maps, residuals, graphical abstract
│   ├── tables/            # Summary statistics and model output CSVs
├── README.md              # The document that you're reading this in. Hope you know that!
├── LICENSE                # MIT License
├── requirements.txt       # Python package dependencies
├── environment.yml        # (optional) Conda environment configuration
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

This project was developed in Google Colab Pro. To run the notebooks:

1. Open any `.ipynb` file in `/notebooks/` in Google Colab
2. Install dependencies in the first cell:

```python
!pip install geopandas mgwr rasterio fiona shapely pyproj


##### Data Sources

- Ambient Air Temperature: CAPA Strategies HeatWatch Campaign
- Tree Canopy Cover: USGS NLCD (2020)
- Elevation (DEM): City of Providence Open Data Portal
- Urban Form Data: City of Providence Open Data Portal
- Green Spaces: City of Providence Open Data Portal & Rhode Island GIS Hub
- Water Bodies:City of Providence Open Data Portal & Rhode Island GIS Hub

All intersected an processed datasets used in modeling are included in the /data/processed/ folder


###### License

This repository is published under the MIT license. See LICENSE for details


###### Acknowledgments

Special thanks to Carolyn, Xana, and Reynolds, who supported this work in ways beyond measure.
This model was developed independently by Kyle Wire and builds upon climate planning research conducted for Brown University.


####### Citations

Please cite this repository as: 

@misc{wire2025heat,
  author       = {Kyle Wire},
  title        = {Modeling Extreme Heat Vulnerability in Providence, RI},
  year         = 2025,
  howpublished = {\url{https://github.com/Kyle-Wire/ModelingExtremeHeat_GWP_ProvidenceRI_050225}},
  note         = {Geographically Weighted Regression, PCA, Urban Climate Modeling}
}


