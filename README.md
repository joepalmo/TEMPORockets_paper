This repository contains the code, data, and figures that accompany the manuscript:  

> **“Quantifying launch vehicle NOx emissions using geostationary satellite observations”**  

We use nitrogen dioxide (NO₂) column measurements from the **Tropospheric Emissions: Monitoring of Pollution (TEMPO)** satellite to detect rocket plumes and derive top-down estimates of nitrogen oxide (NOₓ) emissions. This work provides the first satellite-based quantification of launch vehicle emissions, demonstrating that observed NOₓ emissions exceed existing bottom-up estimates.  

---

## Repository Structure

```
├── setup.py                 # Basic setup file
├── tempo.py                 # TEMPO utilities / data handling
├── analysis.py              # Core analysis routines
├── compute_emissions.py     # Scripts for calculating NOx emissions
├── detect_plumes.py         # Plume detection algorithm
├── emissions.py             # Emission factor calculations
├── download_tempo.py        # Script to fetch TEMPO data
├── make_animation.py        # Code for generating animations
├── utils.py                 # Shared helper functions
├── figures**.ipynb          # Jupyter notebooks for generating paper figures
├── figures/                 # .png files of manuscript figures
├── data/                    # Processed data files
├── plumes/                  # Extracted plume datasets, plume visualizations
├── false_positives/         # Manually verified false detections
├── README.md                # You are here
├── .gitignore               # Ignore rules
```

---

## Key Files & Notebooks

- **Notebooks for figures**  
  - `figure_1.ipynb`, `figure_2_S6.ipynb`, `figure_3.ipynb`, `figure_S1.ipynb`, etc. → Reproduce manuscript figures.  
- **Error analysis**  
  - `error_analysis.ipynb` → Uncertainty quantification.  
- **Analysis scripts**  
  - `detect_plumes.py` → Identifies NO₂ enhancements associated with launches.  
  - `compute_emissions.py` → Converts plume NO₂ enhancements into NOₓ emissions using lifetime and NO/NO₂ ratios.  
  - `emissions.py` → Contains functions for emission calculation.  
  - `analysis.py` → Takes detected plumes and performs initial analysis (calculate plume processing and make time-correction; save data files for visualization)
- **Data handling**  
  - `download_tempo.py` → Downloads raw TEMPO files from NASA Earthdata.  
  - `tempo.py` → Provides utilities for working with TEMPO datasets.  
- **Visualization**  
  - `make_animation.py` → Creates animations of TEMPO NO2 to show false positives.

## Data
- `data/initial_detections.csv` is a table including information on all the detections that are reported in the results.
- **Processed plume datasets** are stored in `plumes/*.nc`.
- **Plume detection visualizations** are stored in `plumes/figs/detections`.  
- **Plume nondetection visualizations** are stored in `plumes/figs/nondetections`.  
- **False positive checks** (launches with interference) are in `false_positives/`.  
- Raw TEMPO Level 2 & 3 data are available from the [NASA Atmospheric Science Data Center](https://asdc.larc.nasa.gov/project/TEMPO).
- Rocket launch data is available from the [General Catalog of Artificial Space Objects](https://planet4589.org/space/gcat/)
