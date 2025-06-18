# Mapping 12 million lakes globally at 10 m resolution
his repository contains a ​UNet-based neural network model​ and supporting code for ​global lake segmentation. The original implementation was developed by Ankit Kariryaa (Kariryaa AT uni-bremen DOT de) in 2018. Subsequent modifications were made by Xuehui Pi and Qiuqi Luo in 2020, followed by further updates by Beihui Hu in 2024. For inquiries, please contact Hu.

## Setup and Installation
See [INSTALL](./INSTALL.md).

## Structure
The code of the three steps is structured in Jupyter notebooks available in the noteooks/ folder. Each notebook is supported with core libraries available in the notebooks/core directory. Input, output paths and other configurations for each notebook must be declared in the notebooks/config/ directory. Please follow the following steps for training a UNet model and for mapping global lakes using the trained UNet model.

### Step 1: Data preparation- [Preprocessing.ipynb](notebooks/1-Preprocessing.ipynb)
There are two main data, the satellite images and the label of lakes in those images. The satellite images which are used to train the model should be annotated with the lakes, while the areas that are annotated should be separately marked and stored as shapefiles. 
The required shapefiles is available in the SampleAnnotations, including the labelled areas and the object polygons (i.e., total_regions.shp and total_polygons.shp). The object polygons shapefiles include a valid id column and no other attribute columns, while the labelled areas shapefiles include a grid id column, file name column, dataset column which presents dataset type (‘train’, ’test’ or ’val’), and the region type column which presents different lake patterns. 
Then, copy notebooks/configTemplate/ directory into notebooks/config/ and declare the input paths and other relevant configurations in notebooks/config/Preprocessing.py file. After declaring the required paths, run the first notebook notebooks/1-Preprocessing.ipynb to extract these areas with the contained object polygons as image files.

### Step 2: Model training - [UNetTraining.ipynb](notebooks/2-UNetTraining.ipynb)
After declaring the relevant configuration in notebooks/config/UNetTraining.py, run the second notebook notebooks/2-UNetTraining.ipynb to train the UNet model with the extracted images. Auxiliary-1-UNetEvaluation.ipynb can be used to evaluate the performance of the model, if you have an independent test set. Step-1 data preparation can also be used to extract the test set.

### Step 3: Analyzing images - [RasterAnalysis.ipynb](notebooks/3-RasterAnalysis.ipynb)
Next, the path to the trained model and satellite images should be declared in the notebooks/config/RasterAnalysis.py, and use the trained model to map global lakes using RasterAnalysis.ipynb notebook. The images to be analyzed can be split into smaller images with Auxiliary-2-SplitRasterToAnalyse.ipynb notebook if the machine doesn't have enough memory to handle large Raster files.
 

