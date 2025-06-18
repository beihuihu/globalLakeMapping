# Author: Ankit Kariryaa, University of Bremen.
# Modified by Xuehui Pi, Qiuqi Luo and Beihui Hu

import os

# Configuration of the parameters for the 1-Preprocessing.ipynb notebook
class Configuration:
    '''
    Configuration for the first notebook.
    Copy the configTemplate folder and define the paths to input and output data. Variables such as raw_NDWI_image_prefix may also need to be corrected if you are use a different source.
    '''
    def __init__(self):
        # For reading the training areas and polygons 
        self.training_base_dir = r''
        self.training_area_fn = os.path.join(self.training_base_dir,r'SampleAnnotations\total_regions.shp')         
        self.training_polygon_fn = os.path.join(self.training_base_dir,r'SampleAnnotations\total_polygons.shp')
        self.type_num=5
        self.dataset_dir = os.path.join(self.training_base_dir,'dataset')

        # For reading images
#         self.bands0 = [0]# If raster has multiple channels, then bands will be [0, 1, ...] otherwise simply [0]
        self.bands = [0,1,2,3,4,5]
        self.raw_image_base_dir =r''
        self.image_type = '.tif'
        self.raw_image_prefix = 'image'
        self.show_boundaries_during_processing = False
        self.ann_type = '.png'
        self.extracted_image_filename = 'image_type'
        self.extracted_annotation_filename = 'annotation_type'
