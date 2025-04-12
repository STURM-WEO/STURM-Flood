# STURM-Flood

This repository hosts the code for the *STURM-Flood* dataset.

The STURM-Flood dataset is a high-quality, open-access resource designed for training and evaluating deep learning models for flood extent mapping using Sentinel-1 and Sentinel-2 satellite imagery. 

The repository is hosted at Zenodo  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12748983.svg)](https://doi.org/10.5281/zenodo.12748983) and provides 21,602 Sentinel-1 tiles and 2,675 Sentinel-2 tiles, each of size 128 × 128 pixels at a resolution of 10 meters, along with corresponding water masks covering 60 flood events globally. 

We invite researchers to utilize this dataset for advancing flood mapping techniques in disaster management. 

For more information about the methodology and results, please refer to our study: [STURM-Flood: a curated dataset for deep learning-basedflood extent mapping leveraging Sentinel-1 and Sentinel-2 imagery](https://doi.org/10.1080/20964471.2025.2458714).

# Tiles Map
Click on the image for an interactive map showing the distribution of Sentinel-1 and Sentinel-2 tiles.

[![View the interactive map](https://github.com/STURM-WEO/STURM-Flood/blob/gh-pages/maps/static.png)](https://sturm-weo.github.io/STURM-Flood/maps/STURM-flood-tiles-map.html)

# Inference Notebook

We provide a Jupyter Notebook for running inference with the pretrained U-Net models on Sentinel-1 and Sentinel-2 imagery. The notebook supports:

- Loading dataset and pretrained models from Zenodo
- Performing flood extent segmentation on single tiles
- Visualizing input and predicted masks
- (Optional) Computing evaluation metrics and visualize contingency maps when ground truth is available

## 🔗 Resources

| Resource        | DOI / Link |
|-----------------|------------|
| **Dataset**     | [10.5281/zenodo.12748982](https://doi.org/10.5281/zenodo.12748982) |
| **Sentinel-1 Model** | [10.5281/zenodo.15189664](https://doi.org/10.5281/zenodo.15189664) |
| **Sentinel-2 Model** | [10.5281/zenodo.15189633](https://doi.org/10.5281/zenodo.15189633) |
| **Published paper**  | [Big Earth Data (2025)](https://doi.org/10.1080/20964471.2025.2458714) |

## 📘 Notebook

Download the notebook here: [`inference_sturm_flood_clean.ipynb`](inference_sturm_flood_clean.ipynb)


## Requirements

- Python ≥ 3.8

- TensorFlow

- NumPy

- Pandas

- Matplotlib

- Rasterio

- Scikit-learn

- requests

- tqdm

- IPython

You can install the necessary libraries using `pip install -r requirements.txt`.

##  Example Outputs
Water probability maps (`*_probs.tif`)

Binary water masks (`*_binary.tif`)

PNG visualizations (input, predicted mask or ground truth mask, contingency map if ground truth is available)

CSV file with per-tile metrics (if ground truth is available)

## Suggested Usage
Set `with_gt = True` to compute and store evaluation metrics.

Set `with_gt = False` to run blind inference when ground truth is not available (e.g. operational scenarios).

# Funding
STURM is an EU-funded R&I project funded under the Marie Skłodowska-Curie Actions (MSCA) Postdoctoral Fellowships - European Fellowships ([Grant agreement ID: 101105589](https://doi.org/10.3030/101105589)) and hosted at [WEO](https://www.weo-water.com/).

# Citation
If you use STURM-Flood or the inference code in your work, please cite:

Notarangelo, N. M., Wirion, C., & van Winsen, F. (2025). STURM-Flood: A curated dataset for deep learning-based flood extent mapping leveraging Sentinel-1 and Sentinel-2 imagery. Big Earth Data. https://doi.org/10.1080/20964471.2025.2458714

# License

This repository is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) License](https://creativecommons.org/licenses/by-sa/4.0/).

The following data sources were used:
- Copernicus EMS ([Manual](https://emergency.copernicus.eu/mapping/sites/default/files/files/JRCTechnicalReport_2020_Manual%20for%20Rapid%20Mapping%20Products_final.pdf))
- Copernicus Sentinel-1 and Sentinel-2 ([License](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice)).
