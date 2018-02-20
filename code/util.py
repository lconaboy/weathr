"""General utility stuff for our project. Put things like regions of
interest here, so that they don't clutter other files, but they're
always accessible when needed.

"""
import glob
import numpy as np
from PIL import Image

def slice_d(region):
    """The difference between region slice's stop and start."""
    start = region.start
    stop = region.stop

    return stop - start

def make_region(slicex=slice(None), slicey=slice(None)):
    """Makes a region with default empty slices."""
    return [slicex, slicey]

def image_region(image, region):
    """Returns the slice of image using region."""
    return image[region[0], region[1]]

def load_images_with_region(files, region):
    """Loads all images in files narrowed to the given region.

Note: be careful, this could raise a MemoryError exception."""
    orig_shape = Image.open(files[0]).size
    images = np.zeros((slice_d(region[0]), slice_d(region[1]), len(files)))

    for idx in np.arange(0, len(files)):
        print('Loading file {} [{}/{}]'.format(files[idx], idx+1, len(files)), end='\r')
        image = np.asarray(Image.open(files[idx]), dtype=int)
        images[:, :, idx] = image_region(image, region) * (1 - land_mask)

    print('All files loaded...', end='\n')
    return images


# Our particular areas of interest. Useful for cutting down on
# processing time, and focusing in on the action.
weathr_regions = {'capetown': make_region(slice(2615, 3015), slice(2350, 2750))}

# These are named paths/globs for data we use. We should at some point come
# up with a better way of storing our data as it's all over the place
# at the moment.
weathr_data = {'vis6': './bands13/vis6/*.jpg',
               'landmask': './landmask.gif'}

land_mask = image_region(np.asarray(Image.open(weathr_data['landmask']), dtype=int),
                         weathr_regions['capetown'])
