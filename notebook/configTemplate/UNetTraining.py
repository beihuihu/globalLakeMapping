# Author: Ankit Kariryaa, University of Bremen.
# Modified by Xuehui Pi, Qiuqi Luo and Beihui Hu

import os

# Configuration of the parameters for the 2-UNetTraining.ipynb notebook
class Configuration:
    def __init__(self):
        # Initialize the data related variables used in the notebook
        # For reading the input images and annotated images generated in the Preprocessing step.
        # In most cases, they will take the same value as in the config/Preprocessing.py
        
        self.base_dir = r''
        self.dataset_dir=os.path.join(self.base_dir,'dataset')
        self.image_type = '.tif'       
        self.ann_type = '.png'
        self.annotation_fn = 'annotation_type'
        self.image_fn = 'image_type'
        self.type_num=4 # The nummber of region types

        # Patch generation; from the training areas (extracted in the last notebook), we generate fixed size patches.
        # random: a random training area is selected and a patch is extracted from a random location inside that training area. Uses a lazy stratergy i.e. batch of patches are extracted on demand.
        # sequential: training areas are selected in the given order and patches extracted from these areas sequential with a given step size. All the possible patches are returned in one call.
        self.patch_generation_stratergy = 'random' # 'random' or 'sequential'    
        self.patch_size = (576,576,6) # Height * Width * (Input or Output) channels  
        self.step_size = (576,576)# # When stratergy == sequential, then you need the step_size as well
        self.input_shape = (576,576,5)
        
        # Shape of the input data, height*width*channel; Here channels are band NDWI, Red, Green, Blue and SWIR
        self.input_image_channel = [0,1,2,3,4]
        self.input_label_channel = [5]

        # CNN model related variables used in the notebook
        self.BATCH_SIZE = 16
        self.NB_EPOCHS = 200

        self.model_path = os.path.join(self.base_dir, 'saved_models') 
        self.steps_per_epoch=691 #steps_per_epoch=(num_train/batch_size)
        self.validation_steps=224 #validation_steps=(num_val/batch_size)