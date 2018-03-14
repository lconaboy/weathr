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
    new_shape = image_region(np.asarray(Image.open(files[0]), dtype=int), region).shape
    images = np.zeros((new_shape[0], new_shape[1], len(files)))

    for idx in np.arange(0, len(files)):
        print('Loading file {} [{}/{}]'.format(files[idx], idx+1, len(files)), end='\r')
        image = np.asarray(Image.open(files[idx]), dtype=int)
        images[:, :, idx] = image_region(image, region) * (1 - image_region(land_mask, region))

    print('All files loaded...', end='\n')
    return images

def path_to_weathr_data(year='08', band='vis6'):
    path = './data/'
    path_year = path + year +'/'
    path_band = path_year + band +'/*.jpg'

    return path_band

def sep_months(fnames, year, band):
    months = np.zeros(len(fnames), dtype=int)
    path = path_to_weathr_data(year, band)
    x0 = len(path[:-5]) + 28
    x1 = x0 + 2
    for i in range(0, len(fnames)):
        months[i] = int(fnames[i][x0:x1])

    return months


# Our particular areas of interest. Useful for cutting down on
# processing time, and focusing in on the action.
weathr_regions = {
    'all': make_region(),
    'africa': make_region(slice(620, 3020), slice(1260, 3000)),
    'capetown': make_region(slice(2615, 3015), slice(2350, 2750))}

# These are named paths/globs for data we use. We should at some point come
# up with a better way of storing our data as it's all over the place
# at the moment.
weathr_data = {'vis6': './bands13/vis6/*.jpg',
               'landmask': './landmask.gif'}

# land_mask = image_region(np.asarray(Image.open(weathr_data['landmask']), dtype=int),
#                           weathr_regions['capetown'])
land_mask = np.asarray(Image.open(weathr_data['landmask']), dtype=int)
