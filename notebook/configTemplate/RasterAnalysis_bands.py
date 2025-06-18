# Author: Ankit Kariryaa, University of Bremen.
# Modified by Xuehui Pi, Qiuqi Luo and Beihui Hu

import os

# Configuration of the parameters for the 3-FinalRasterAnalysis.ipynb notebook
class Configuration:
    '''
    Configuration for the notebook where objects are predicted in the image.
    Copy the configTemplate folder and define the paths to input and output data.
    '''
    def __init__(self):
        self.input_image_dir = r''
        self.input_image_type = '.tif'
        self.image_fn_st = 'image'
      
        self.type_num=5
        self.band_num=5

        self.base_dir = r''
        self.model_path = os.path.join(self.base_dir, 'saved_models') 
        self.trained_model_path = os.path.join(self.model_path,'lakes_20240130-2243_AdaDelta_dice_loss_012345_576.h5')
        print('self.trained_model_path:', self.trained_model_path)
        
        self.output_image_type = '.tif'
        self.output_dir = r''
        self.output_prefix = 'pre'  
        self.output_shapefile_type = '.shp'
        self.overwrite_analysed_files =False
        self.output_dtype='uint8'

        # Variables related to batches and model
        self.ignore_edge_width=100
        self.BATCH_SIZE =16# Depends upon GPU memory and WIDTH and HEIGHT (Note: Batch_size for prediction can be different then for training.
        self.WIDTH=576# Should be same as the WIDTH used for training the model
        self.HEIGHT=576 # Should be same as the HEIGHT used for training the model
        self.STRIDE=576-2*self.ignore_edge_width