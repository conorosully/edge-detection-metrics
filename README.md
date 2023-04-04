# edge-detection-metrics
Exploring the effectiveness of RMSE, PSNR, SSIM and FOM for evaluation edge detection for coastline extraction 

This repository contains the code required to reproduce the results in the conference paper:

> To update

This code is only for academic and research purposes. Please cite the above paper if you intend to use whole/part of the code. 

## Data Files

We have used the following dataset in our analysis: 

1. Sentinel-2 Water Edges Dataset (SWED) from [UK Hydrographic Office](https://openmldata.ukho.gov.uk/#:~:text=The%20Sentinel%2D2%20Water%20Edges,required%20for%20the%20segmentation%20mask.).

 The data is available under the Geospatial Commission Data Exploration license.

## Code Files
You can find the following files in the src folder:

- `comparison-metrics.ipynb` The main analysis file used to apply Canny edge detection, calculate evaluation metrics and create all figures in the research paper. The file is also used to display the figures used to perform the visual analysis. 
- `utils.py` Helper file containing functions used to perform the analysis in the main analysis file. 
- `test-image-issues.ipynb` Display the images in the SWED test set that had erroneous segmentation masks

## Result Files
You can find the following files used in the analysis:

- `Visual Analysis.xlsx` Contains the results of the visual analysis
- `canny_evaluation_metrics.csv` Contains the values for RMSE, PSNR, SSIM and FOM and confusion matrix measures at different hysteresis thresholds
