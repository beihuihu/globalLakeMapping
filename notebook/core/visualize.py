#   Author: Ankit Kariryaa, University of Bremen
#   Modified by Beihui Hu
import matplotlib.pyplot as plt  # plotting tools
import numpy as np
import earthpy.plot as ep

def display_images(img, fn=None,titles=None,cmap=None, norm=None, interpolation=None):
    """Display the given set of images, optionally with titles.
    images: array of image tensors in Batch * Height * Width * Channel format.
    titles: optional. A list of titles to display with each image.
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    cols = img.shape[-1]-2
    rows = img.shape[0]

    plt.figure(figsize=(20, 20 * rows // cols),dpi=150)
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.05,hspace=0.05)
    for i in range(rows):
        for j in range(cols):
            ax=plt.subplot(rows, cols, (i*cols) + j + 1 )
            ax.set_axis_off()
            ## for subplots in first row, add titles
            if (i==0) and (titles!=None):
                ax.set_title(titles[j],fontsize=12)
            # band ndwi
            if j==0:
                plt.imshow(img[i,...,j], cmap=cmap, norm=norm, interpolation=interpolation)
            # band Red, Green, Blue
            elif j==1:
                rgb=np.transpose(img[i,...,j:j+3], axes=(2,0,1))
                ep.plot_rgb(rgb,ax=ax, stretch=True,str_clip=0.5)

            #Band SWIR/annotation/prediction
            else:
                plt.imshow(img[i,...,j+2], cmap=cmap, norm=norm, interpolation=interpolation)

    if fn!=None:
        plt.savefig(fn,bbox_inches='tight',pad_inches = 0.1)
        plt.close()
    
